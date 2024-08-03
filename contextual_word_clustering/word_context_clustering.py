from os import listdir
from os.path import isfile, join, dirname
import json
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import mpld3

#list with word and sentence to extract contextual embeddings from
with open(join(dirname(__file__), 'sentence_to_cluster.json'), 'r') as f:
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

    sentence_dict[sent_index] = {"sentence": sentence_text, "contextual_embedding": contextual_embedding, "word": word}
    embeddings.append(contextual_embedding)

#apply tsne for 3d projection
tsne_embeddings = TSNE(n_components=2).fit_transform(np.array(embeddings))

#Agglomerative Hierarchichal Clustering with Ward's method
linkage_data = linkage(tsne_embeddings, method='ward', metric='euclidean')
dendrogram(linkage_data)

#save clustering as png
plt.title("Word: " + word)
plt.savefig(join(dirname(__file__), 'ward_clustering.png'))

#get clusters
hc = AgglomerativeClustering(n_clusters = 25, metric = 'euclidean', linkage ='ward')
embedding_labels = hc.fit_predict(tsne_embeddings)

#save data in a json file
for sent_index in range(1, len(sentence_list)):
    sentence_dict[sent_index]["tsne_embedding"] = tsne_embeddings[sent_index - 1].tolist()
    sentence_dict[sent_index]["cluster_label"] = int(embedding_labels[sent_index - 1])

with open(join(dirname(__file__), 'word_embedding_data.json'), 'w') as f:
    json.dump(sentence_dict, f)

#plot points, with color representing cluster
fig, ax = plt.subplots()
plt.title("Word: " + word)
fig.set_size_inches(10, 10)

unzip = [[i for [i, j] in tsne_embeddings],
       [j for [i, j] in tsne_embeddings]]

scatter = ax.scatter(unzip[0], unzip[1], c=embedding_labels)

#plot in browser with sentence appearing when hovering
labels = [sentence_dict[i]["sentence"] for i in range(1, len(sentence_list))]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()

#save plot as a png
plt.savefig(join(dirname(__file__), 'word_context_clustering.png'))