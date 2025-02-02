import numpy as np
from os import listdir
from os.path import isfile, join, dirname
from itertools import islice
import json
import plotly.express as px

with open(join(dirname(__file__), 'selected_sentences.json'), 'r') as f:
    sentences = json.load(f)

categ_count = {}
for sentence in sentences:
    if sentence["subfolder"] not in categ_count:
        categ_count[sentence["subfolder"]] = 0
    categ_count[sentence["subfolder"]] += 1

labels = categ_count.keys()
values = categ_count.values()

fig = px.pie(names=labels, values=values, title="Source of sentences")
fig.write_html(join(dirname(__file__), 'sent_categ_pie.html'))
fig.show()
