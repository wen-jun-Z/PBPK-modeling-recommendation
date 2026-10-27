"""Resolve article-level links from stored provenance, without network or LLM calls."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote, urlsplit, parse_qs

from sqlalchemy import text
from lxml import etree, html

ROOT = Path(__file__).resolve().parents[3]
MAX_PDF_BYTES = 64 * 1024 * 1024


def http_url(value):
    value = str(value or '').strip()
    match = re.fullmatch(r'\[[^\]]*\]\((https?://.+)\)', value)
    if match:
        value = match[1].replace('\\&', '&')
    try:
        parsed = urlsplit(value)
        if parsed.scheme in ('https', 'http') and parsed.hostname and not parsed.username and not parsed.password:
            return value
    except ValueError:
        pass
    return None


def blueprint_evidence_url(content: bytes, evidence: dict, compound_name: str, compound_id: str) -> str | None:
    """Resolve the selected evidence, never the first URL in a multi-drug file.

    The caller verifies the source bytes. Only explicit document identifiers,
    selected records and parameter-level provenance are used; no network or
    filesystem access, and no search through unrelated references.
    """
    parameter = evidence.get('parameter', '')
    location = str(evidence.get('source', {}).get('location', ''))
    expected_cid = compound_id.removeprefix('pubchem:') if compound_id.startswith('pubchem:') else ''

    def page(value):
        url = http_url(value)
        if not url:
            return None
        parsed = urlsplit(url)
        if parsed.hostname == 'pubchem.ncbi.nlm.nih.gov':
            match = re.search(r'/(?:compound/(?:cid/)?|data/compound/)(\d+)(?:/|$)', parsed.path)
            cid = match[1] if match else expected_cid if '/compound/name/' in parsed.path else ''
            if cid.isdigit():
                return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + cid
        return url

    def direct(record):
        if not isinstance(record, dict):
            return None
        for key in ('source_url', 'url', 'report_url'):
            if found := page(record.get(key)):
                return found
        source = record.get('source')
        if isinstance(source, dict):
            return page(source.get('url') or source.get('source_url'))
        return None

    for record in (evidence.get('context'), evidence.get('source')):
        if found := direct(record):
            return found
    try:
        data = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        try:
            document = etree.fromstring(content, parser=etree.XMLParser(resolve_entities=False, no_network=True))
            # Only the document's own metadata; never article IDs in its references.
            if etree.QName(document).localname == 'article':
                for kind, prefix in (('pmc', 'https://pmc.ncbi.nlm.nih.gov/articles/'),
                                     ('pmcid', 'https://pmc.ncbi.nlm.nih.gov/articles/'),
                                     ('pmid', 'https://pubmed.ncbi.nlm.nih.gov/'),
                                     ('doi', 'https://doi.org/')):
                    ids = document.xpath('./front/article-meta/article-id[@pub-id-type=$kind]/text()', kind=kind)
                    if len(ids) == 1:
                        value = ids[0].strip()
                        if kind in ('pmc', 'pmcid') and re.fullmatch(r'(?:PMC)?\d+', value):
                            return prefix + (value if value.startswith('PMC') else 'PMC'+value) + '/'
                        if kind == 'pmid' and value.isdigit():
                            return prefix + value + '/'
                        if kind == 'doi' and re.fullmatch(r'10\.\d{4,9}/\S+', value):
                            return prefix + quote(value, safe='/():;')
            if document.tag == '{urn:hl7-org:v3}document':
                ids = document.xpath('./h:setId/@root', namespaces={'h': 'urn:hl7-org:v3'})
                if len(ids) == 1 and re.fullmatch(r'[0-9a-fA-F-]{36}', ids[0]):
                    return 'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=' + ids[0].lower()
        except (etree.XMLSyntaxError, ValueError):
            pass
        try:
            document = html.fromstring(content, parser=html.HTMLParser(no_network=True))
            canonical = document.xpath('//head/link[contains(concat(" ", normalize-space(@rel), " "), " canonical ")]/@href')
            if len(canonical) == 1 and (found := page(canonical[0])):
                return found
            ids = document.xpath('//head/meta[translate(@name,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="documentsetid"]/@content')
            if len(ids) == 1 and re.fullmatch(r'[0-9a-fA-F-]{36}', ids[0]):
                return 'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=' + ids[0].lower()
        except (etree.ParserError, ValueError):
            pass
        return None
    if not isinstance(data, dict):
        return None

    if isinstance(data.get('records'), dict):
        match = re.match(r'/records/([^/]+)', location)
        key = match[1].replace('~1', '/').replace('~0', '~') if match else compound_name
        records = [record for name, record in data['records'].items() if name.casefold() == key.casefold()]
        if len(records) != 1 or not isinstance(records[0], dict):
            return None
        data = records[0]
    if isinstance(data.get('cases'), list):
        case_id = re.split(r'[:/]', location, maxsplit=1)[0]
        cases = [case for case in data['cases'] if case.get('case_id') == case_id]
        if len(cases) != 1:
            return None
        entries = cases[0].get('evidence', {})
        key = {'total_clearance': 'total_clearance_ml_min', 'renal_clearance': 'renal_clearance_ml_min'}.get(parameter, parameter)
        if parameter == 'elimination_scope':
            links = [direct(entries.get(k)) for k in ('total_clearance_ml_min', 'renal_clearance_ml_min')]
            return links[0] if all(links) and len(set(links)) == 1 else None
        return direct(entries.get(key))

    selected = data.get('selected_physicochemical_inputs')
    if isinstance(selected, dict):
        key = 'acidic_pka' if parameter == 'pka_types' and 'pka_types' not in selected else parameter
        entry = selected.get(key, {})
        if found := direct(entry):
            return found
        identity = data.get('identity', {})
        cid = str(identity.get('pubchem_cid', ''))
        if parameter in ('molecular_weight', 'molecular_formula') and cid.isdigit():
            return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + cid
        # Explicit source descriptions connect these older curated inputs to pKa/XLogP.
        term = r'pka' if parameter == 'pka_types' else r'xlogp|logp' if parameter == 'logp' else None
        sources = [s for s in data.get('source_files', []) if term and re.search(term, str(s.get('location', '')), re.I)]
        links = {url for s in sources if (url := direct(s))}
        if len(links) == 1:
            return links.pop()
        if cid.isdigit() and (any('pubchem' in str(s.get('location', '')).lower() for s in sources)
                             or 'pubchem' in str(entry.get('note', '')).lower()):
            return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + cid
        return None

    response = data.get('response', data)
    if not isinstance(response, dict):
        return None
    results = response.get('resultList', {}).get('result')
    if isinstance(results, list):
        match = re.search(r'result\[id=(\d+)\]', location)
        index = re.search(r'result\[(\d+)\]', location)
        if match:
            results = [r for r in results if str(r.get('id')) == match[1]]
        elif index:
            results = results[int(index[1]):int(index[1])+1]
        if len(results) != 1:
            return None
        record = results[0]
        pmid = str(record.get('pmid') or (record.get('id') if record.get('source') == 'MED' else '') or '')
        if pmid.isdigit():
            return 'https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/'
        doi = str(record.get('doi') or '')
        return 'https://doi.org/' + quote(doi, safe='/():;') if re.fullmatch(r'10\.\d{4,9}/\S+', doi) else None
    record = response.get('Record', {})
    if record.get('RecordType') == 'CID' and str(record.get('RecordNumber', '')).isdigit():
        return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + str(record['RecordNumber'])
    properties = response.get('PropertyTable', {}).get('Properties', [])
    if isinstance(properties, list):
        matches = [p for p in properties if str(p.get('CID', '')) == expected_cid]
        if len(matches) == 1 and expected_cid.isdigit():
            return 'https://pubchem.ncbi.nlm.nih.gov/compound/' + expected_cid
    study = re.fullmatch(r'pmid:(\d+)', str(data.get('study_id', '')), re.I)
    if study:
        return 'https://pubmed.ncbi.nlm.nih.gov/' + study[1] + '/'
    return direct(data) or direct(data.get('primary_model_source'))


def identifiers(document):
    record = document.get('parsed_record') or {}
    metadata = record.get('metadata_json') or {}
    pmid = str(metadata.get('pmid') or '')
    pmcid = str(metadata.get('pmcid') or '').upper()
    doi = str(metadata.get('doi') or '').strip()
    if pmcid.isdigit():
        pmcid = 'PMC' + pmcid
    return {
        'pmid': pmid if re.fullmatch(r'\d+', pmid) else None,
        'pmcid': pmcid if re.fullmatch(r'PMC\d+', pmcid) else None,
        'doi': doi if re.fullmatch(r'10\.\d{4,9}/\S+', doi) else None,
    }


def article_links(document, normalized=None):
    """Never infer a PMID by picking an ID from a batch efetch request."""
    ids = identifiers(document)
    source = document.get('source_id')
    url = http_url(document.get('source_url'))
    links = []

    def add(label, href, kind):
        if href and not any(x['url'] == href for x in links):
            links.append({'label': label, 'url': href, 'kind': kind})

    if ids['pmid']:
        add('PubMed 文章', f"https://pubmed.ncbi.nlm.nih.gov/{ids['pmid']}/", 'article')
    if ids['pmcid']:
        add('PMC 全文', f"https://pmc.ncbi.nlm.nih.gov/articles/{ids['pmcid']}/", 'full_text')
    if ids['doi']:
        add('期刊原文 / DOI', 'https://doi.org/' + quote(ids['doi'], safe='/():;'), 'publisher')
    if url and source == 'dailymed':
        match = re.search(r'/spls/([0-9a-f-]{36})(?:\.|/|$)', url, re.I)
        setid = match[1] if match else parse_qs(urlsplit(url).query).get('setid', [''])[0]
        if re.fullmatch(r'[0-9a-f-]{36}', setid, re.I):
            add('DailyMed 说明书', 'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid='+setid, 'article')
    cid = str((normalized or {}).get('pubchem_cid') or '')
    if source == 'pubchem' and cid.isdigit():
        add('PubChem 化合物', f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}', 'article')
    # Keep only an actual page/PDF as fallback, not JSON/XML APIs or OA tarballs.
    if url and not links:
        parsed = urlsplit(url)
        if parsed.path.lower().endswith('.pdf'):
            add('来源 PDF', url, 'pdf_remote')
        elif not (parsed.hostname == 'eutils.ncbi.nlm.nih.gov' or
                  re.search(r'\.(?:xml|json|tgz|tar\.gz|tar_gz)$', parsed.path, re.I) or
                  '/rest/' in parsed.path or '/services/' in parsed.path):
            add('来源网页', url, 'article')
    return links


@lru_cache(maxsize=2)
def _pdf_index(root, time_bucket):
    index = {}
    for path in (root / 'artifacts/pbpk_raw_runs').rglob('*.pdf'):
        index.setdefault(path.name, []).append(path)
    return index


def _pdf_candidates(basename, root):
    # One cached filename scan for all requests, with a five-minute refresh.
    if Path(basename).name != basename or not basename.lower().endswith('.pdf'):
        return ()
    return tuple(_pdf_index(root, int(time.monotonic() // 300)).get(basename, ()))


@lru_cache(maxsize=256)
def _verified_pdf(path, expected, size, mtime):
    if not re.fullmatch('[a-fA-F0-9]{64}', expected or '') or size > MAX_PDF_BYTES:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        if b'%PDF-' not in stream.read(1024):
            return False
        stream.seek(0)
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected.lower()


def resolve_pdf(record, root=ROOT):
    value = str(record.get('raw_storage_path') or '')
    if not value.lower().endswith('.pdf'):
        return None
    path = Path(value)
    # Check boundary before any stat/read. Symlinks escaping the project are rejected.
    direct = path if path.is_absolute() else root / path
    candidates = [direct] if direct.is_relative_to(root / 'artifacts') else []
    candidates.extend(_pdf_candidates(path.name, root))
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root / 'artifacts'):
            continue
        try:
            stat = candidate.stat()
            if candidate.is_file() and _verified_pdf(candidate, record.get('raw_checksum'), stat.st_size, stat.st_mtime_ns):
                return candidate
        except OSError:
            continue
    return None


def document_provenance(db, document_id=None, *, evidence=None):
    evidence = evidence or {}
    fields = evidence.get('extracted_fields') or {}
    summary = evidence.get('summary') or {}
    document_id = document_id or fields.get('parsed_document_id')
    document = None
    if document_id:
        document = db.execute(text('''SELECT p.parsed_document_id::text,p.source_id,p.title,
            p.document_type,p.publication_date,p.source_location,p.parsed_record,
            r.raw_document_id::text,r.source_url,r.raw_storage_path,r.raw_checksum
            FROM parsed_document p LEFT JOIN raw_document r USING(raw_document_id)
            WHERE p.parsed_document_id=:id'''), {'id': document_id}).mappings().first()
    document = dict(document or evidence.get('source_document') or {})
    if not document:
        document = {'source_id': summary.get('source_id'), 'source_url': fields.get('source_url'),
                    'title': summary.get('parameter_name'), 'source_location': summary.get('source_location')}
    ids = identifiers(document)
    links = article_links(document, evidence.get('normalized_fields'))
    pdf_records = []
    if str(document.get('raw_storage_path') or '').lower().endswith('.pdf'):
        pdf_records.append(document)
    if ids['pmcid']:
        # Metadata-bound exact article ID, not title similarity or the retrieval drug.
        rows = db.execute(text('''SELECT raw_document_id::text,source_url,raw_storage_path,raw_checksum
            FROM raw_document WHERE (raw_storage_path ILIKE '%.pdf' OR raw_mime_type='application/pdf')
            AND (source_url ILIKE :pattern OR raw_storage_path ILIKE :pattern)
            ORDER BY raw_document_id'''), {'pattern': '%'+ids['pmcid']+'%'}).mappings()
        for row in rows:
            if re.search(r'(?<![A-Za-z0-9])'+ids['pmcid']+r'(?!\d)', str(row['source_url'])+' '+str(row['raw_storage_path']), re.I):
                pdf_records.append(dict(row))
    pdf_records.sort(key=lambda r: bool(re.search(r'moesm|supplement|appendix|_esm', str(r.get('source_url') or ''), re.I)))
    for record in pdf_records:
        path = resolve_pdf(record)
        if path:
            links.append({'label': '打开 PDF', 'url': '/api/workspace/source-pdf/'+record['raw_document_id'],
                          'kind': 'pdf_local', 'verification': 'sha256_matches_registered_source'})
            break
        url = http_url(record.get('source_url'))
        if url and urlsplit(url).path.lower().endswith('.pdf') and not any(x['kind'].startswith('pdf') for x in links):
            links.append({'label': '来源 PDF', 'url': url, 'kind': 'pdf_remote'})
    has_pdf = any(x['kind'].startswith('pdf') for x in links)
    return {'title': document.get('title') or '来源资料', 'source': document.get('source_id'),
            **ids, 'publication_date': str(document['publication_date']) if document.get('publication_date') else None,
            'location': summary.get('source_location') or document.get('source_location'),
            'links': links, 'pdf_note': None if has_pdf else '未收录可核验的 PDF；可从全文或期刊页面查看可用版本。',
            'link_basis': 'stored_article_metadata_and_checksum_verified_local_assets'}
