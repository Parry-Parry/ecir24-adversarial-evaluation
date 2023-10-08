import pandas as pd 
from fire import Fire 

METRICS = ['MRC', 'Success Rate']
MODEL_DICT = {'t5.small' : r'monoT5$_\text{small}$', 't5.base' : r'monoT5$_\text{base}$', 't5.large' : r'monoT5$_\text{large}$', 't5.3b' : r'monoT5$_\text{3B}$'}

DATA_DICT = {'dl19' : 'DL19', 'dl20' : 'DL20'}

TOKEN_GROUPS = {
    'Prompt Tokens' : ['true', 'false', 'relevant', 'relevanttrue', 'relevantfalse'],
    'Control Tokens' : ['bar', 'baz', 'information', 'informationbar', 'informationbaz', 'relevantbar', 'informationtrue'],
    'Synonyms' : ['pertinent', 'significant', 'related', 'associated', 'important'],
    'Sub-Words' : ['relevancy', 'relevance', 'relevantly','irrelevant'],
    'Misspellings' : ['relevanty', 'relevent', 'trues', 'falses']
}

POSITIONS = {
    'start' : 's',
    'end' : 'e',
    'random' : 'r'
}

def format_colour(value, colour_level, percent=False):
    if percent:
        if value >= 0.5:
            return r'\cellcolor{' + 'pos' + f'!{colour_level}' + '}' f'${abs(value)}' + r'\%$'
        else:
            return r'\cellcolor{' + 'neg' + f'!{colour_level}' + '}' + f'${abs(value)}' + r'\%$'
        
    if value < 0.: return r'\cellcolor{' + 'neg' + f'!{colour_level}' + '}' + f'$-{abs(value)}$'
    elif value==0.: return r'\cellcolor{pos!0}' + f'$+{abs(value)}$'
    else: return r'\cellcolor{' + 'pos' + f'!{colour_level}' + '}' f'$+{abs(value)}$'

def format_mrc(mrc, sr, colour_level, sig):
    sig_tok = r'\sig' if sig else r'\insig'
    colour_token = 'pos' if mrc >= 0. else 'neg'
    sign = '+' if mrc >= 0. else '-'
    if mrc == 0.: sign = ''
    out = r'\cellcolor{' + colour_token + f'!{colour_level}' + '}' + f'${sign}{abs(mrc)}' + sig_tok + '_{\color{gray}\,\phantom{0}' + f'{abs(sr)}' + r'\%}$' 
    return out

def main(run_file : str, out_file : str):
    df = pd.read_csv(run_file, sep='\t', index_col=False)
    # for each metric find the absolute maximum value 
    tmp = df[df.model != 'bm25'].copy()
    max_vals = {
        #'MSC' : max(df[df.metric=='MSC'].value.max(), abs(df[df.metric=='MSC'].value.min())),
        'MRC' : tmp[tmp.metric=='MRC'].value.max(),
        'Success Rate' : tmp[tmp.metric=='Success Rate'].value.max()
    }

    print(max_vals)
    
    def colour_combo(mrc, sr, sig=False):
        max_val = float(max_vals['MRC'])
        abs_val = abs(mrc)
        # min max normalise abs_val between max val and 0 
        norm_val = (abs_val - 0) / (max_val - 0)
        norm_val = round(norm_val * 50)
        norm_val = min(norm_val, 50)
        return format_mrc(round(mrc, 1), round(sr * 100, 1), norm_val, sig)
    
    preamble = r'\begin{tabular}{@{}lrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr@{}}'
    header = r'\toprule'
    columns = 'Token & ' + ' & '.join([r'\multicolumn{' + str(len(DATA_DICT)*3) + r'}{c}{' + f'{model}' + r'}' for _, model in MODEL_DICT.items()]) + r'\\'    
    # for each model column write each dataset from data_dict twice 
    datasets = '& ' + ' & '.join([' & '.join(r'\multicolumn{3}{c}{' + f'{data}' + r'}' for _, data in DATA_DICT.items())] * len(MODEL_DICT)) + r'\\'
    metrics = '& ' + ' & '.join([' & '.join(['P', 'R', 'MRC SR'] * len(DATA_DICT))] * len(MODEL_DICT)) + r'\\'
    total = [preamble, header, columns, r'\midrule', datasets, r'\midrule', metrics, r'\midrule']
    for group, tokens in TOKEN_GROUPS.items():
        total.append(r'\midrule')
        total.append(r'\multicolumn{13}{l}{' + group + r'}\\')
        total.append(r'\midrule')
        for token in tokens:
            row = ''
            token_subset = df[df.token==token].copy()
            row += token + ' & '
            for val, _ in MODEL_DICT.items():
                model_subset = token_subset[token_subset.model==val].copy()
                print(model_subset)
                assert len(model_subset) == len(DATA_DICT) * 3
                for data, _ in DATA_DICT.items():
                    data_subset = model_subset[model_subset.dataset==data].copy()
                    mrc = float(data_subset[data_subset.metric=='MRC'].value.values[0])
                    sr = float(data_subset[data_subset.metric=='Success Rate'].value.values[0])
                    sig = bool(data_subset[data_subset.metric=='sig'].value.values[0])
                    row += POSITIONS[data_subset.position.values[0]] + ' & '
                    row += str(data_subset.n_tok.values[0]) + ' & '
                    row += colour_combo(mrc, sr, sig) + ' & '
            row = row[:-2] + r'\\'
            total.append(row)
    total.append(r'\bottomrule')
    total.append(r'\label{tab:transfer}')
    total.append(r'\end{tabular}')
    with open(out_file, 'w') as f:
        f.write('\n'.join(total))

if __name__ == '__main__':
    Fire(main)
        