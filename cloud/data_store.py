"""Load only the portable, hash-verified research snapshot committed with the app."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / 'data'
MAX_BYTES = 32 * 1024 * 1024


@lru_cache(maxsize=4)
def _verified(name: str, digest: str, size: int):
    if name not in {'catalog.json', 'blueprints.json'}:
        raise ValueError('Unknown snapshot file')
    path = DATA / name
    if path.is_symlink() or not 0 < size <= MAX_BYTES or path.stat().st_size != size:
        raise ValueError('Snapshot size does not match the manifest')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Snapshot checksum failed; redeploy the complete committed bundle')
    data = json.loads(raw)
    if data.get('metadata', {}).get('schema_version') != 1:
        raise ValueError('Unsupported snapshot schema version')
    return data


def _load(name: str):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported snapshot manifest version')
    entry = manifest['files'][name]
    return _verified(name, entry['sha256'], entry['bytes'])


def load_catalog():
    return _load('catalog.json')


def load_blueprints():
    return _load('blueprints.json')
