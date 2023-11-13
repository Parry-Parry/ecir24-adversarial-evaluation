#!/usr/bin/env python3
import pyterrier as pt
import pandas as pd
pt.start()
from pyterrier_t5 import MonoT5ReRanker
monoT5 = MonoT5ReRanker()

df = pd.read_json('t5-base-re-ranking.json.gz', lines=True)
#monoT5 = MonoT5ReRanker()
df = df.head(100)

df = pt.Transformer.from_df(df)
