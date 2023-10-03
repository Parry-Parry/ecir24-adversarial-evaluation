# ECIR Context Agnostic Ranking Attacks
Experiments for Seq2Seq Ranking Attacks

## Notes

### data/bm25_19.tsv.gz

gzip compressed DL-19 results for BM25 

File is a tsv (sep='\t')

Columns:
* qid : string, query_id
* query : string, query text
* docno : string, doc_id
* score : float, bm25 score
* rank : integer, automated terrier rank
* text : string, document text
