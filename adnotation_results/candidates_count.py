import json
from os import listdir
from os.path import isfile, join, dirname
import plotly.express as px
import pandas as pd

with open(join(dirname(__file__), 'fluency.jsonl'), 'r') as f:
    fluency = [json.loads(line) for line in f]

with open(join(dirname(__file__), 'simplicity.json'), 'r') as f:
    simplicity = json.load(f)

adnotations = fluency + simplicity

romanian_lcp = pd.read_csv(join(dirname(__file__), 'romanian_lcp.tsv'), sep='\t', header=None)

candidates = {}

for adn in adnotations:
    id = adn['id']
    word_pair = adn['word_pair']
    if id not in candidates:
        candidates[id] = set([])
    candidates[id].add(word_pair[0])
    candidates[id].add(word_pair[1])

for i in range(romanian_lcp.shape[0]):
    id = romanian_lcp.iloc[i, 0]
    if id not in candidates:
        candidates[id] = set([])
    candidates[id].add(romanian_lcp.iloc[i, 3])

cand_count = {}

for id in candidates:
    if len(candidates[id]) not in cand_count:
        cand_count[len(candidates[id])] = 0
    cand_count[len(candidates[id])] += 1

labels = cand_count.keys()
values = cand_count.values()

fig = px.pie(names=labels, values=values, title="Number of candidates per sentence")
fig.write_html(join(dirname(__file__), 'candidates_count_pie.html'))
fig.show()