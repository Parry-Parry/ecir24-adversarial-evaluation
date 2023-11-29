from fire import Fire
import pandas as pd
import pyterrier as pt
if not pt.started():
    pt.init()

def prune(tokens : str, out_file : str, subset : int = 50, stopwords : bool = True, stopword_subset : int = None, alnum : bool = True):
    if stopwords and stopword_subset is None: stopword_subset = subset // 2
    stopword = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword

    tokens = pd.read_json(tokens, lines=True)
    filter_term = r'\u2581'
    tokens['text'] = tokens['text'].apply(lambda x : x.replace(filter_term, ''))
    print(tokens.head())
    if stopwords:
        main_tokens = tokens[~tokens['text'].apply(lambda x : stopword(x))]
        stopwords = tokens[tokens['text'].apply(lambda x : stopword(x))]
        main_tokens = main_tokens.sort_values('score', ascending=False).head(subset - stopword_subset)
        stopwords = stopwords.sort_values('score', ascending=False).head(stopword_subset)
        main_tokens['type'] = 'non-stop'
        stopwords['type'] = 'stop'
        final_set = pd.concat([main_tokens, stopwords])
    else:
        final_set = tokens.sort_values('score', ascending=False).head(subset)
        final_set['type'] = 'NA'
    final_set.to_json(out_file, lines=True, orient='records')

    return "Done!"

if __name__ == '__main__':
    Fire(prune)


