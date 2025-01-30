import pandas as pd
from os import listdir
from os.path import isfile, join, dirname
import spacy
import numpy as np
import json
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from wordfreq import zipf_frequency
from sklearn.preprocessing import MinMaxScaler

#read data
romanian_lcp = pd.read_csv(join(dirname(__file__), 'romanian_lcp_2.tsv'), sep='\t', header=None)

NLP = spacy.load("ro_core_news_md")

def complexity_data(sentence, word):
    isInSent = False
    
    #get tokens from sentence
    tokens = NLP(sentence)

    #get zipf score for both the word and other words in sentences in relation to our word
    zipf_word = zipf_frequency(word, 'ro')
    zipf_sum = 0.0
    zipf_higher = 0

    for token in tokens:
        zipf_cur = zipf_frequency(token.text, 'ro')
        zipf_sum += zipf_cur
            
        if zipf_cur > zipf_word:
            zipf_higher += 1

        #extract embeddings for the word
        if token.text == word:
            contextual_embedding = token.vector
            isInSent = True

    zipf_avg = zipf_sum / len(tokens)
    zipf_dif = abs(zipf_sum - zipf_avg)
        
    zipf_bin = 0
    if zipf_word < 4.25:
        zipf_bin = 1

    #if word was found add to data
    if isInSent:
        X_zip = [zipf_word, zipf_sum, zipf_dif, zipf_bin]

    return (isInSent, X_zip, contextual_embedding)

data = {}

X_train_emb = []
X_train_zipf = []
y_train = []

#get training data
for i in range(romanian_lcp.shape[0]):
    sentence = romanian_lcp.loc[i][2].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    word = romanian_lcp.loc[i][3]
    score = romanian_lcp.loc[i][4]
    data[i] = {"sentence": sentence, "word": word, "score": score}

    (isInSent, zip, contextual_embedding) = complexity_data(sentence, word)

    #if word was found add to data
    if isInSent:
        data[i] = {"sentence": sentence, "word": word, "score": score, "embedding": contextual_embedding}
        X_train_zipf.append(zip)
        X_train_emb.append(contextual_embedding)
        y_train.append(score)

train_len = len(X_train_emb)

X_test_emb = []
X_test_zipf = []
y_test = []

with open(join(dirname(__file__), 'word_to_selected_sentence.json'), 'r') as f:
    word_to_sentence = json.load(f)

#get test data
for word in word_to_sentence:
    for sent in word_to_sentence[word]:
        (isInSent, zip, emb) = complexity_data(sent["sentence"], word)
        X_test_zipf.append(zip)
        X_test_emb.append(emb)

X_zipf = X_train_zipf + X_test_zipf
X_emb = X_train_emb + X_test_emb

#normalise data
X_tsne = TSNE(n_components=3).fit_transform(np.array(X_emb))
X_tsne_scaled = MinMaxScaler().fit_transform(X_tsne)

X_scaled = MinMaxScaler().fit_transform(X_zipf)

X = []

for i in range(len(X_tsne)):
    X.append(X_tsne[i].tolist() + X_zipf[i])

X_train = X[:train_len]

#model
reg = LinearRegression(fit_intercept=False)

model = reg.fit(X_train, y_train)

#get results
X_test = X[train_len:]
y_pred = model.predict(X_test)

complexity_scores = {}
i = 0
for word in word_to_sentence:
    sum_score = 0
    nr_scores = 0
    for sent in word_to_sentence[word]:
        sum_score += y_pred[i]
        nr_scores += 1
        i += 1
    complexity_scores[word] = sum_score / nr_scores

with open(join(dirname(__file__), 'complexity_scores.json'), 'w') as f:
    json.dump(complexity_scores, f)