import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join, dirname
from itertools import islice
import json
import plotly.express as px

#plot number of sentences per word
with open(join(dirname(__file__), 'word_to_selected_sentence.json'), 'r') as f:
    word_to_sentence = json.load(f)

dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
for word in word_to_sentence:
    dist[len(word_to_sentence[word])] += 1

dist_dict = {"Number of sentences per word": [], "Number of words": []}
for i in range(len(dist)):
    if dist[i] > 0:
        dist_dict["Number of sentences per word"].append(i)
        dist_dict["Number of words"].append(dist[i])

fig = px.bar(dist_dict, x="Number of sentences per word", y="Number of words")
fig.write_html(join(dirname(__file__), 'sentences_per_word_distribution.html'))
fig.show()

#plot complexity distribution
with open(join(dirname(__file__), 'complexity_scores.json'), 'r') as f:
    complexity_scores = json.load(f)

scores = []
for word in complexity_scores:
    scores.append(complexity_scores[word])

fig = px.histogram(x=scores, title="Complexity scores distribution")
fig.write_html(join(dirname(__file__), 'complexity_scores_distribution.html'))
fig.show()