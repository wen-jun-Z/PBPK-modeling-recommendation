from pathlib import Path

from pbpk_kb.recommendation.model_matching.parallel_training import build_jobs


def test_parallel_training_builds_complete_unique_grid(tmp_path: Path) -> None:
    jobs = build_jobs(
        variants=["gmn", "gmn_comparison_fixture"],
        seeds=[17, 29],
        jobs_root=tmp_path / "jobs",
        legacy_main_root=tmp_path / "legacy",
    )

    assert len(jobs) == 4
    assert len({job.job_id for job in jobs}) == 4
    assert jobs[0].output_directory == tmp_path / "legacy" / "seed_17"
    assert jobs[-1].output_directory == (
        tmp_path / "jobs" / "gmn_comparison_fixture" / "seed_29"
    )
