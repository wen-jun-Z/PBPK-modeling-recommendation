from pathlib import Path

import pytest

from pbpk_kb.model_runtime.result_parser import compare_csv_files


HEADER = '#Project: Trial\n#PK-Sim version: 12.3.173\n"IndividualId","Gender","Weight [kg]"\n'


def compare(tmp_path: Path, a: str, b: str):
    left, right = tmp_path / "a.csv", tmp_path / "b.csv"
    left.write_text(a)
    right.write_text(b)
    return compare_csv_files(run_a_path=left, run_b_path=right, simulation_id="trial",
                             csv_ordinal=1, absolute_tolerance=1e-12, relative_tolerance=1e-10)


def test_population_metadata_and_every_numeric_cell_are_compared(tmp_path):
    result = compare(tmp_path, HEADER + '0,MALE,70\n1,FEMALE,60\n',
                     HEADER + '0,MALE,70.00000000001\n1,FEMALE,60\n')
    assert result.accepted
    assert result.metadata_row_count == 2
    assert result.column_count == 3
    assert result.row_count == 2
    assert result.numeric_cell_count == 4
    assert result.non_numeric_cell_count == 2


@pytest.mark.parametrize("changed,reason", [
    (HEADER.replace("12.3.173", "12.4.0") + "0,MALE,70\n", "CSV_METADATA_MISMATCH"),
    (HEADER.split("\n", 1)[1] + "0,MALE,70\n", "CSV_METADATA_MISMATCH"),
    (HEADER + "0,MALE,71\n", "NUMERIC_TOLERANCE_EXCEEDED"),
    (HEADER + "0,MALE\n", "CSV_COLUMN_COUNT_MISMATCH"),
    (HEADER + "0,FEMALE,70\n", "NON_NUMERIC_CELL_MISMATCH"),
    (HEADER + "0,MALE,70\n1,MALE,80\n", "CSV_ROW_COUNT_MISMATCH"),
])
def test_population_format_support_does_not_ignore_mismatches(tmp_path, changed, reason):
    result = compare(tmp_path, HEADER + "0,MALE,70\n", changed)
    assert not result.accepted
    assert reason in {m["reason_code"] for m in result.mismatches}


def test_comment_only_file_is_not_a_table(tmp_path):
    result = compare(tmp_path, "#Project: Trial\n", "#Project: Trial\n")
    assert not result.accepted
    assert result.mismatches[0]["reason_code"] == "CSV_EMPTY"
