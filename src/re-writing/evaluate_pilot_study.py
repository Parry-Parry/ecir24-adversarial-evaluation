#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('pilot-study.csv')
df = df[df['Iter'] == 1]
del df['Iter']

df_chatgpt = df[df['Prompt Model'] == 'chatgpt']
df_chatgpt = df_chatgpt[['Prompt Model', 'Prompt', 'MRC', 'MSC', 'Success Rate']].groupby('Prompt').agg({'Prompt Model': 'first', 'MRC': 'mean', 'MSC': 'mean', 'Success Rate': 'mean'}).reset_index().sort_values('Success Rate', ascending=False)

df_chatgpt.to_csv('chatgpt-pilot-study-selection.csv')

print(df_chatgpt)


df_alpacca = df[df['Prompt Model'] == 'alpacca']
df_alpacca = df_alpacca[['Prompt Model', 'Prompt', 'MRC', 'MSC', 'Success Rate']].groupby('Prompt').agg({'Prompt Model': 'first', 'MRC': 'mean', 'MSC': 'mean', 'Success Rate': 'mean'}).reset_index().sort_values('Success Rate', ascending=False)

df_alpacca.to_csv('alpacca-pilot-study-selection.csv')

print(df_alpacca)
