import bibtexparser

def load_doi_map(bibfile_path):
    """
    Parse the .bib file and return a dict mapping each DOI (normalized to lowercase)
    to its entry key.
    """
    with open(bibfile_path, encoding='utf-8') as bf:
        db = bibtexparser.load(bf)

    doi_map = {}
    for entry in db.entries:
        doi = entry.get('doi')
        if doi:
            doi_map[doi.strip().lower()] = entry['ID']
    return doi_map

def lookup_doi(doi, doi_map):
    """
    Given a DOI (string) and a doi_map from load_doi_map, return the BibTeX key,
    or None if not found.
    """
    return doi_map.get(doi.strip().lower())

from doi_to_key import load_doi_map, lookup_doi
doi_map = load_doi_map('bibliography.bib')
#lookup_doi('10.1016/S0377-2217(98)00113-1', doi_map)