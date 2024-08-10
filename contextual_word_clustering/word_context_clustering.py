from os import listdir
from os.path import isfile, join, dirname
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import plotly.express as px

#list with word and sentence to extract contextual embeddings from
with open(join(dirname(__file__), 'sentence_to_cluster_2.json'), 'r') as f:
    sentence_list = json.load(f)

#word is first element from list
word = sentence_list[0]

#dictionary to track sentences, embeddings and the word
sentence_dict = {}
embeddings = []

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
model = AutoModel.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

#contextualized word embeddings for each sentence
for sent_index in range(1, len(sentence_list)):
    #replace diacritics with prefered version for the romanian bert 
    sentence_text = sentence_list[sent_index].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")

    #get tokens from sentence
    token_ids = tokenizer.encode(sentence_text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    #search word in tokens
    word_index = tokens.index(word)

    #run through model and extract embeddings for the word
    outputs = model(torch.tensor(token_ids).unsqueeze(0))
    contextual_embedding = outputs[0][0][word_index].tolist()

    sentence_dict[sent_index] = {"sentence": sentence_text, "word": word}
    embeddings.append(contextual_embedding)

#apply tsne for 3d projection
tsne_embeddings = TSNE(n_components=2).fit_transform(np.array(embeddings))

#Agglomerative Hierarchichal Clustering with Ward's method
linkage_data = linkage(tsne_embeddings, method='ward', metric='euclidean')
dendrogram(linkage_data)

#save clustering as png
plt.title("Word: " + word)
plt.savefig(join(dirname(__file__), 'ward_clustering.png'))

#get number of clusters with best silhouette score or very close to the best score
kmeans_per_k = [KMeans(n_clusters=k).fit(tsne_embeddings)
                for k in range(2, 21)]
silhouette_scores = [silhouette_score(tsne_embeddings, model.labels_)
                     for model in kmeans_per_k]
nr_clusters = np.argmax(silhouette_scores)
nr_best_clusters = nr_clusters

for i in range(nr_clusters + 1, 19):
    if silhouette_scores[i] > silhouette_scores[nr_clusters] * 0.95:
        nr_best_clusters = i

nr_best_clusters += 2

#get cluster labels and distance to center for each point
kmeans = KMeans(n_clusters = nr_best_clusters).fit(tsne_embeddings)
embedding_labels = kmeans.labels_
kmeans_transform = kmeans.transform(tsne_embeddings)
center_distance = [kmeans_transform[i][embedding_labels[i]] for i in range(len(embedding_labels))]

#save data in a json file
for sent_index in range(1, len(sentence_list)):
    sentence_dict[sent_index]["tsne_embedding"] = tsne_embeddings[sent_index - 1].tolist()
    sentence_dict[sent_index]["cluster_label"] = int(embedding_labels[sent_index - 1])
    sentence_dict[sent_index]["center_distance"] = float(center_distance[sent_index - 1])

#custom sort for sentences based on distance to center of cluster
def distance_key(e):
    return sentence_dict[e]["center_distance"]

subsamples = {}

#extract subsamples from each cluster (15% of sentences , half close to center, half far)
for label in range(nr_best_clusters):
    points = [x for x in sentence_dict if sentence_dict[x]["cluster_label"] == label]
    points.sort(key = distance_key)
    
    nr_points = len(points) * 15 // 100
    nr_points_close = max(nr_points // 2 + nr_points % 2, 1)
    nr_points_far = max(nr_points // 2, 1)

    points_close = points[:nr_points_close]
    points_far = points[-nr_points_far:]
    subsamples[label] = {"close_to_center": [sentence_dict[x] for x in points_close], "far_from_center": [sentence_dict[x] for x in points_far]}

#save data to json file
with open(join(dirname(__file__), 'subsamples_data_2.json'), 'w') as f:
    json.dump(subsamples, f)

#unzip data for plotting
unzip = [[i for [i, j] in tsne_embeddings],
       [j for [i, j] in tsne_embeddings]]

#plot in html with sentence and data appearing when hovering
labels = [sentence_dict[i]["sentence"] for i in range(1, len(sentence_list))]   

fig_px = px.scatter(x=unzip[0], y=unzip[1], color=embedding_labels, hover_name=labels, title="Word: " + word)

#save plot as html
fig_px.write_html(join(dirname(__file__), 'word_context_clustering_2.html'))
fig_px.show()