# ECIR Context Agnostic Ranking Attacks
Experiments for potential ECIR submission attacking neural ranking models

## Notes

### data/bm25_19.tsv.gz

gzip compressed DL-19 results for BM25 

File is a tsv (sep='\t')

Columns:
* qid : string, query_id
* query : string, query text
* docno : string, doc_id
* score : float, bm25 score
* text : string, document text
