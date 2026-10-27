"""Finish frozen rankings while complete-model execution is repaired.

This entry point cannot score clinical outcomes. It uses the existing bounded
worker supervisor and original input/code checks; all score files stay in place.
"""
from pathlib import Path
import fcntl
import json
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_recommendation_execution as driver


def main():
    run = driver.RUN / 'ranking_continuation'
    run.mkdir(exist_ok=True)
    with (driver.RUN / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if driver.alive():
            raise RuntimeError('Original pipeline must be stopped before isolated ranking continuation')
        frozen = driver.freeze_preparation()
        supervisor = driver.Supervisor(frozen)
        def stop(*_):
            supervisor.stop_event.set()
            (driver.RUN / 'stop.requested').touch()
        signal.signal(signal.SIGTERM, stop)
        driver.save(run / 'status.json', {'state':'running_rankings_only', 'started_at':driver.now(),
                                        'clinical_scoring_permitted':False})
        try:
            supervisor.rankings(during_preparation=True)
            driver.save(run / 'status.json', {'state':'rankings_complete', 'finished_at':driver.now(),
                                             'clinical_scoring_permitted':False})
        except BaseException as error:
            driver.save(run / 'status.json', {'state':'failed', 'finished_at':driver.now(),
                                             'error':str(error)[:400], 'clinical_scoring_permitted':False})
            raise


if __name__ == '__main__':
    main()
