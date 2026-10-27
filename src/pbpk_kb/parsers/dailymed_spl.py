from __future__ import annotations

from typing import Any
from xml.etree import ElementTree as ET


class DailyMedSPLParser:
    spl_namespace = {"hl7": "urn:hl7-org:v3"}
    clinical_pharmacology_code = "34090-1"
    pk_section_codes = {"43682-4"}
    ddi_section_codes = {"34073-7"}
    pk_title_keywords = {
        "PHARMACOKINETIC",
        "ABSORPTION",
        "DISTRIBUTION",
        "METABOLISM",
        "ELIMINATION",
        "BIOAVAILABILITY",
        "SPECIAL POPULATIONS",
        "RENAL",
        "HEPATIC",
    }
    ddi_title_keywords = {
        "DRUG INTERACTIONS",
        "INTERACTIONS",
        "CONTRAINDICATIONS",
        "PRECAUTIONS",
    }

    def parse(self, xml_payload: bytes) -> dict[str, Any]:
        root = ET.fromstring(xml_payload)
        sections = self._extract_sections(root)
        return {
            "active_ingredients": self._extract_active_ingredients(root),
            "active_moieties": self._extract_active_moieties(root),
            "clinical_pharmacology_section": self._select_clinical_pharmacology_section(sections),
            "pk_section_candidates": self._select_section_candidates(
                sections,
                codes=self.pk_section_codes,
                keywords=self.pk_title_keywords,
            ),
            "ddi_section_candidates": self._select_section_candidates(
                sections,
                codes=self.ddi_section_codes,
                keywords=self.ddi_title_keywords,
            ),
            "section_count": len(sections),
        }

    def _extract_active_ingredients(self, root: ET.Element) -> list[dict[str, Any]]:
        ingredients: list[dict[str, Any]] = []
        seen: set[tuple[str | None, str | None]] = set()
        for ingredient in root.findall(".//hl7:ingredient", self.spl_namespace):
            class_code = ingredient.get("classCode")
            if class_code not in {"ACTIM", "ACTIB"}:
                continue
            substance = ingredient.find("hl7:ingredientSubstance", self.spl_namespace)
            if substance is None:
                continue
            name = self._normalize_text(substance.findtext("hl7:name", default="", namespaces=self.spl_namespace))
            code_element = substance.find("hl7:code", self.spl_namespace)
            ingredient_code = code_element.get("code") if code_element is not None else None
            key = (name, ingredient_code)
            if not name or key in seen:
                continue
            seen.add(key)
            numerator = ingredient.find("hl7:quantity/hl7:numerator", self.spl_namespace)
            denominator = ingredient.find("hl7:quantity/hl7:denominator", self.spl_namespace)
            ingredients.append(
                {
                    "name": name,
                    "class_code": class_code,
                    "unii": ingredient_code,
                    "strength_numerator_value": numerator.get("value") if numerator is not None else None,
                    "strength_numerator_unit": numerator.get("unit") if numerator is not None else None,
                    "strength_denominator_value": denominator.get("value") if denominator is not None else None,
                    "strength_denominator_unit": denominator.get("unit") if denominator is not None else None,
                }
            )
        return ingredients

    def _extract_active_moieties(self, root: ET.Element) -> list[dict[str, str | None]]:
        moieties: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str | None]] = set()
        for moiety in root.findall(".//hl7:activeMoiety/hl7:activeMoiety", self.spl_namespace):
            name = self._normalize_text(moiety.findtext("hl7:name", default="", namespaces=self.spl_namespace))
            code_element = moiety.find("hl7:code", self.spl_namespace)
            moiety_code = code_element.get("code") if code_element is not None else None
            key = (name, moiety_code)
            if not name or key in seen:
                continue
            seen.add(key)
            moieties.append({"name": name, "unii": moiety_code})
        return moieties

    def _extract_sections(self, root: ET.Element) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        for index, section in enumerate(root.findall(".//hl7:section", self.spl_namespace), start=1):
            code_element = section.find("hl7:code", self.spl_namespace)
            title = self._normalize_text(section.findtext("hl7:title", default="", namespaces=self.spl_namespace))
            code = code_element.get("code") if code_element is not None else None
            display_name = code_element.get("displayName") if code_element is not None else None
            text = self._normalize_text(" ".join(section.itertext()))
            sections.append(
                {
                    "section_index": index,
                    "title": title,
                    "code": code,
                    "display_name": display_name,
                    "text": text,
                    "text_char_count": len(text),
                    "source_location": f"SECTION:{code or index}:{title or display_name or 'untitled'}",
                }
            )
        return sections

    def _select_clinical_pharmacology_section(self, sections: list[dict[str, Any]]) -> dict[str, Any] | None:
        for section in sections:
            title = str(section.get("title") or "").upper()
            display_name = str(section.get("display_name") or "").upper()
            if (
                section.get("code") == self.clinical_pharmacology_code
                or "CLINICAL PHARMACOLOGY" in title
                or "CLINICAL PHARMACOLOGY" in display_name
            ):
                return section
        return None

    @staticmethod
    def _select_section_candidates(
        sections: list[dict[str, Any]],
        *,
        codes: set[str],
        keywords: set[str],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for section in sections:
            title = str(section.get("title") or "").upper()
            display_name = str(section.get("display_name") or "").upper()
            code = section.get("code")
            candidate_reason: str | None = None
            if code in codes:
                candidate_reason = f"section_code:{code}"
            elif any(keyword in title or keyword in display_name for keyword in keywords):
                candidate_reason = "title_or_display_name_keyword"
            if candidate_reason:
                candidate = dict(section)
                candidate["candidate_reason"] = candidate_reason
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _normalize_text(text: str | None) -> str:
        return " ".join((text or "").split())
