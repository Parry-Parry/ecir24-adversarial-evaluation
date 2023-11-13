#!/usr/bin/env python3
import pandas as pd
from fire import Fire
from os.path import join

def main(study_file : str, out_dir : str):

    df = pd.read_csv(study_file)
    df = df[df['Iter'] == 1]
    del df['Iter']

    df_chatgpt = df[df['Prompt Model'] == 'chatgpt']
    df_chatgpt = df_chatgpt[['Prompt Model', 'Prompt', 'MRC', 'MSC', 'Success Rate']].groupby('Prompt').agg({'Prompt Model': 'first', 'MRC': 'mean', 'MSC': 'mean', 'Success Rate': 'mean'}).reset_index().sort_values('Success Rate', ascending=False)

    df_chatgpt.to_csv(join(out_dir, 'chatgpt-pilot-study-selection.csv'))

    print(df_chatgpt)


    df_alpacca = df[df['Prompt Model'] == 'alpacca']
    df_alpacca = df_alpacca[['Prompt Model', 'Prompt', 'MRC', 'MSC', 'Success Rate']].groupby('Prompt').agg({'Prompt Model': 'first', 'MRC': 'mean', 'MSC': 'mean', 'Success Rate': 'mean'}).reset_index().sort_values('Success Rate', ascending=False)

    df_alpacca.to_csv(join(out_dir, 'alpacca-pilot-study-selection.csv'))

    print(df_alpacca)

if __name__ == '__main__':
    Fire(main)