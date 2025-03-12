import json
import pandas as pd
import numpy as np
from wordfreq import zipf_frequency
from scipy.stats import pearsonr

with open('labeling_scores.jsonl', 'r', encoding='utf-8') as fin:
  fis = fin.readlines()


items = []
for line in fis:
  js = json.loads(line)
  scrs = {
    "id": js['id'],
    'text': js['text'],
    'word': js['meta']['word'],
    'mean': js['mean_score'],
    'frq': zipf_frequency(js['meta']['word'].lower().strip(), lang='ro'),
    'var': js['variance']
    }  
  for user, score in zip(js['labels'], js['scores']):
    scrs[user[3]] = score
  vals = np.array(js['scores'])
  comp = np.sum(vals>=0.5)
  simp = np.sum(vals<=0.5)
  marj = 'complex' if comp > simp else 'simple'
  if comp == simp:
    marj = 'inconclusive'
  scrs['majority'] = marj
  items.append(scrs)

df = pd.DataFrame(items)
print(df.majority.value_counts())

print(pearsonr(df.frq, df['mean']))


#print(pearsonr(df.frq, df[['victor', 'sergiu', 'petru', 'stadio88', 'user', 'iulia']].mean(axis=1)))


idx = df.frq>0.00001
df2 = df[idx]

print(pearsonr(df2.frq, df2['mean']))

#usrs = ['victor', 'sergiu', 'petru', 'stadio88']
#print(pearsonr(df2.frq, df2[['victor', 'sergiu', 'petru', 'stadio88', 'user', 'iulia']].mean(axis=1)))

df.to_csv('labeling_scores.csv', index=False)

'''
{"id": 5348, "text": "Adaug\u0103 un buton la bara de instrumente de editare din editorul de surse, astfel \u00eenc\u00e2t este cel mai convenabil pentru paginile de discu\u021bii ale utilizatorilor cu leg\u0103turi ro\u0219ii, dar poate economisi un clic sau dou\u0103 \u0219i \u00een alte cazuri, chiar dac\u0103 nu este la fel de rapid ca Twinkle.", "meta": {"word": "leg\u0103turi", "en_id": "en_247", "en_wd": "redlinked"}, "labels": [[160, 169, "simplu", "victor"], [160, 168, "complex", "sergiu"], [160, 168, "foarte familiar", "petru"], [160, 174, "nu cunosc", "stadio88"]], "scores": [0.25, 0.75, 0.0, 1.0], "mean_score": 0.5, "variance": 0.15625}
'''