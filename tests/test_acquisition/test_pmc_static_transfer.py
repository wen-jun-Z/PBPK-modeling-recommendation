from pathlib import Path
from subprocess import CompletedProcess

from pbpk_kb.acquisition.transfer_pmc_static_objects import (
    MAX_RANGE_REQUEST_BYTES,
    PMCObjectTransfer,
    content_priority,
    fixed_byte_ranges,
    object_key,
    object_key_from_asset,
    split_byte_ranges,
    storage_path,
)


def test_pmc_s3_object_identity_is_stable_between_plan_and_asset(tmp_path: Path) -> None:
    plan = {
        "pmcid": "PMC123",
        "s3_key": "PMC123.1/PMC123.1.pdf",
        "url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/PMC123.1.pdf",
    }
    asset = {"url": plan["url"], "metadata": {"s3_key": plan["s3_key"]}}

    assert object_key(plan) == "s3:PMC123.1/PMC123.1.pdf"
    assert object_key_from_asset(asset) == object_key(plan)


def test_pmc_static_storage_is_outside_compound_specific_paths(tmp_path: Path) -> None:
    plan = {
        "pmcid": "PMC123",
        "s3_key": "PMC123.1/PMC123.1.xml",
        "url": "https://pmc-oa-opendata.s3.amazonaws.com/PMC123.1/PMC123.1.xml",
    }

    path = storage_path(tmp_path, plan)

    assert path.parent == tmp_path / "shared" / "pmc_static_objects" / "PMC123" / "pmc_oa"
    assert path.name.endswith(".xml")


def test_pmc_text_and_pdf_precede_incidental_images() -> None:
    xml = {"s3_key": "PMC123.1/PMC123.1.xml"}
    pdf = {"s3_key": "PMC123.1/PMC123.1.pdf"}
    image = {"s3_key": "PMC123.1/figure.jpg"}

    assert content_priority(xml) < content_priority(pdf) < content_priority(image)


def test_parallel_ranges_cover_only_the_missing_suffix() -> None:
    ranges = split_byte_ranges(
        start_offset=100,
        expected_bytes=1_100,
        connections=4,
        min_segment_bytes=1,
    )

    assert [(item.start, item.end) for item in ranges] == [
        (100, 349),
        (350, 599),
        (600, 849),
        (850, 1_099),
    ]
    assert sum(item.byte_count for item in ranges) == 1_000


def test_fixed_byte_ranges_cover_an_object_in_one_mebibyte_chunks() -> None:
    ranges = fixed_byte_ranges(2 * MAX_RANGE_REQUEST_BYTES + 1)

    assert [(item.start, item.end) for item in ranges] == [
        (0, MAX_RANGE_REQUEST_BYTES - 1),
        (MAX_RANGE_REQUEST_BYTES, 2 * MAX_RANGE_REQUEST_BYTES - 1),
        (2 * MAX_RANGE_REQUEST_BYTES, 2 * MAX_RANGE_REQUEST_BYTES),
    ]


def test_range_download_resumes_after_a_short_successful_response(tmp_path: Path) -> None:
    transfer = PMCObjectTransfer(
        run_root=tmp_path,
        workers=1,
        connect_timeout_seconds=1,
        low_speed_limit_bytes=1,
        low_speed_time_seconds=1,
        max_transfer_seconds=1,
        retries=0,
        range_connections=1,
        range_min_bytes=1,
        max_connections=1,
        dry_run=False,
        max_objects=None,
    )
    calls: list[str] = []

    def write_three_bytes(command: list[str]) -> CompletedProcess[str]:
        range_value = command[command.index("--range") + 1]
        start, end = (int(value) for value in range_value.split("-"))
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(output.read_bytes() + b"x" * min(3, end - start + 1) if output.exists() else b"x" * min(3, end - start + 1))
        calls.append(range_value)
        return CompletedProcess(command, 0, "", "")

    transfer.run_curl = write_three_bytes  # type: ignore[method-assign]
    path = tmp_path / "segment.part.range00"
    transfer.download_range("https://example.invalid/object", {"path": str(path), "start": 10, "end": 19})

    assert path.read_bytes() == b"x" * 10
    assert calls == ["10-19", "13-19", "16-19", "19-19"]


def test_range_download_retries_a_zero_byte_successful_response(tmp_path: Path) -> None:
    transfer = PMCObjectTransfer(
        run_root=tmp_path,
        workers=1,
        connect_timeout_seconds=1,
        low_speed_limit_bytes=1,
        low_speed_time_seconds=1,
        max_transfer_seconds=1,
        retries=0,
        range_connections=1,
        range_min_bytes=1,
        max_connections=1,
        dry_run=False,
        max_objects=None,
    )
    outcomes = iter([3, 0, 3, 3, 1])

    def occasionally_empty(command: list[str]) -> CompletedProcess[str]:
        range_value = command[command.index("--range") + 1]
        start, end = (int(value) for value in range_value.split("-"))
        byte_count = min(next(outcomes), end - start + 1)
        output = Path(command[command.index("--output") + 1])
        if byte_count:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(output.read_bytes() + b"x" * byte_count if output.exists() else b"x" * byte_count)
        return CompletedProcess(command, 0, "", "")

    transfer.run_curl = occasionally_empty  # type: ignore[method-assign]
    path = tmp_path / "segment.part.range00"
    transfer.download_range("https://example.invalid/object", {"path": str(path), "start": 0, "end": 9})

    assert path.read_bytes() == b"x" * 10


def test_range_download_limits_each_request_to_one_mebibyte(tmp_path: Path) -> None:
    transfer = PMCObjectTransfer(
        run_root=tmp_path,
        workers=1,
        connect_timeout_seconds=1,
        low_speed_limit_bytes=1,
        low_speed_time_seconds=1,
        max_transfer_seconds=1,
        retries=0,
        range_connections=1,
        range_min_bytes=1,
        max_connections=1,
        dry_run=False,
        max_objects=None,
    )
    calls: list[str] = []

    def write_requested_bytes(command: list[str]) -> CompletedProcess[str]:
        range_value = command[command.index("--range") + 1]
        start, end = (int(value) for value in range_value.split("-"))
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        data = b"x" * (end - start + 1)
        output.write_bytes(output.read_bytes() + data if output.exists() else data)
        calls.append(range_value)
        return CompletedProcess(command, 0, "", "")

    transfer.run_curl = write_requested_bytes  # type: ignore[method-assign]
    path = tmp_path / "segment.part.range00"
    transfer.download_range(
        "https://example.invalid/object",
        {"path": str(path), "start": 0, "end": 2 * 1024 * 1024},
    )

    assert path.stat().st_size == 2 * 1024 * 1024 + 1
    assert calls == ["0-1048575", "1048576-2097151", "2097152-2097152"]


def test_verified_chunks_are_assembled_and_cleaned_up(tmp_path: Path) -> None:
    transfer = PMCObjectTransfer(
        run_root=tmp_path,
        workers=1,
        connect_timeout_seconds=1,
        low_speed_limit_bytes=1,
        low_speed_time_seconds=1,
        max_transfer_seconds=1,
        retries=0,
        range_connections=1,
        range_min_bytes=1,
        max_connections=1,
        dry_run=False,
        max_objects=None,
    )

    def write_requested_chunk(command: list[str]) -> CompletedProcess[str]:
        range_value = command[command.index("--range") + 1]
        start, end = (int(value) for value in range_value.split("-"))
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(bytes([start // MAX_RANGE_REQUEST_BYTES]) * (end - start + 1))
        return CompletedProcess(command, 0, "", "")

    transfer.run_curl = write_requested_chunk  # type: ignore[method-assign]
    temporary = tmp_path / "object.bin.part"
    expected = 2 * MAX_RANGE_REQUEST_BYTES + 1
    transfer.download_with_verified_chunks("https://example.invalid/object", temporary, expected)

    assert temporary.read_bytes() == (
        b"\x00" * MAX_RANGE_REQUEST_BYTES
        + b"\x01" * MAX_RANGE_REQUEST_BYTES
        + b"\x02"
    )
    assert not list(tmp_path.glob("object.bin.part.chunk*"))
