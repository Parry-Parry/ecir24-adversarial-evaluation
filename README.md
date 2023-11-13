# Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models

Experiments for evaluating Seq2Seq Ranking Attacks

## Usage

TODO:
- Installation
- Dependencies
- Mining
- Injection
- Re-writing
- Evaluation

## Package Structure
```
advseq2seq
├───directory-processing: Compression and join scripts for retrieval effectiveness calculation
├───re-writing: LLM rewrite functions for processing set of documents
│   └───evaluation: Evaluate rewritten documents
├───retrieval_effectiveness: Evaluate retrieval effectiveness
└───stuffing: Keyword Stuffing
    ├───evaluation: Evaluation over re-rankers
    ├───injection: Keyword stuffing from some token file
    └───table_generation: Latex table generation
```

## Data Structure
```
data
├───intermediate: Splits of BM25 scored documents for efficient API querying
├───llm-rewrite
│   ├───bm25_19_in_progress: Intermediate prompt runs for efficient API querying
│   ├───bm25_20_in_progress: Intermediate prompt runs for efficient API querying
│   ├───pilot-study: All runs used to determine best prompts for LLM rewrite
│   │   └───selection: Evaluation of runs
│   ├───raw-api-chatgpt: Raw API output from openai
│   └───test-output: Output from tests of evaluation setup
├───rewriting-runs: All runs scored by each re-ranker over LLM rewritten documents
│   └───trec-runs
│       ├───dl19
│       └───dl20
├───search-provider
└───stuffing-runs: All runs scored by each re-ranker over keyword stuffed documents
    ├───dl19
    └───dl20
```