"""Article identity and private PDF boundary checks; no external requests."""
import hashlib
from pathlib import Path

import pytest

from pbpk_kb.web.provenance import article_links, identifiers, http_url, resolve_pdf


def doc(metadata, url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=8695020,10197295'):
    return {'source_id':'pubmed','source_url':url,'parsed_record':{'metadata_json':metadata}}


def test_pubmed_batch_uses_article_metadata_not_first_batch_id():
    links=article_links(doc({'pmid':'10197295','doi':'10.1177/00912709922007930'}))
    assert links[0]['url']=='https://pubmed.ncbi.nlm.nih.gov/10197295/'
    assert links[1]['url']=='https://doi.org/10.1177/00912709922007930'
    assert all('8695020' not in x['url'] and not x['kind'].startswith('pdf') for x in links)


def test_missing_pubmed_id_does_not_guess_from_batch():
    assert article_links(doc({}))==[]


@pytest.mark.parametrize('raw',['javascript:alert(1)','file:///etc/passwd','https://user:password@example.org/doc.pdf','//example.org'])
def test_unsafe_link_rejected(raw):
    assert http_url(raw) is None


def test_pmc_and_dailymed_links():
    assert identifiers(doc({'pmcid':'12345'}))['pmcid']=='PMC12345'
    assert article_links(doc({'pmcid':'12345'}))[0]['url']=='https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/'
    x={'source_id':'dailymed','source_url':'https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/f051776f-94c5-4e34-80ed-f45b9c36881e.xml'}
    assert article_links(x)[0]['url'].endswith('drugInfo.cfm?setid=f051776f-94c5-4e34-80ed-f45b9c36881e')


def test_verified_migrated_pdf_and_tampering(tmp_path):
    folder=tmp_path/'artifacts/pbpk_raw_runs/batch';folder.mkdir(parents=True)
    path=folder/'article.pdf';data=b'%PDF-1.4\npublic article';path.write_bytes(data)
    record={'raw_storage_path':'/Volumes/old/article.pdf','raw_checksum':hashlib.sha256(data).hexdigest()}
    assert resolve_pdf(record,tmp_path)==path
    path.write_bytes(b'%PDF-1.4\ntampered different content')
    assert resolve_pdf(record,tmp_path) is None


def test_unregistered_hash_and_symlink_escape_rejected(tmp_path):
    folder=tmp_path/'artifacts/pbpk_raw_runs';folder.mkdir(parents=True)
    outside=tmp_path/'private.pdf';outside.write_bytes(b'%PDF-private')
    (folder/'private.pdf').symlink_to(outside)
    record={'raw_storage_path':str(outside),'raw_checksum':hashlib.sha256(outside.read_bytes()).hexdigest()}
    assert resolve_pdf(record,tmp_path) is None
    pdf=folder/'missing-hash.pdf';pdf.write_bytes(b'%PDF-unverified')
    assert resolve_pdf({'raw_storage_path':str(pdf)},tmp_path) is None
