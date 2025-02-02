from os import listdir
from os.path import isfile, join, dirname
from itertools import islice
import json
import spacy
import re

with open(join(dirname(__file__), 'word_to_sentence_filtered.json'), 'r') as f:
    sentences = json.load(f)

with open(join(dirname(__file__), 'file_subfolder.json'), 'r') as f:
    file_subfolder = json.load(f)

words_set = set([])
selected_sentences = []
word_to_selected_sentence = {}
sent_dict = {}

for i in range(5, 0, -1):
    i_key = str(i)
    for key in sentences[i_key]:
        words = re.findall(r'\w+', key)
        count = sum([1 for word in words if word in words_set])
        if count <= i / 2:
            sentence =  sentences[i_key][key][0]["sentence"].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
            selected_sentences.append({"sentence": sentence, "words": words, "file": sentences[i_key][key][0]["file"], "subfolder": file_subfolder[sentences[i_key][key][0]["file"]]})
            sent_dict[sentence] = {"file": sentences[i_key][key][0]["file"], "subfolder": file_subfolder[sentences[i_key][key][0]["file"]]}
            for word in words:
                if word not in words_set:
                    words_set.add(word)
                    word_to_selected_sentence[word] = [{"sentence": sentence, "file": sentences[i_key][key][0]["file"], "subfolder": file_subfolder[sentences[i_key][key][0]["file"]]}]
                else:
                    word_to_selected_sentence[word].append({"sentence": sentence, "file": sentences[i_key][key][0]["file"], "subfolder": file_subfolder[sentences[i_key][key][0]["file"]]})

with open(join(dirname(__file__), 'selected_words_subsample_data.json'), 'r') as f:
    subsamples = json.load(f)

for word in words_set:
    if word in subsamples:
        new_selected_sentences = []
        for subsample in subsamples[word]["subsamples"]:
            sent_data = {"sentence": subsamples[word]["subsamples"][subsample]["close_to_center"][0]["sentence"]}
            if sent_data["sentence"] in sent_dict:
                sent_data["file"] = sent_dict[sent_data["sentence"]]["file"]
                sent_data["subfolder"] = sent_dict[sent_data["sentence"]]["subfolder"]
            new_selected_sentences.append(sent_data)
        word_to_selected_sentence[word] = new_selected_sentences

print(len(selected_sentences))
print(len(word_to_selected_sentence))

with open(join(dirname(__file__), 'selected_sentences.json'), 'w') as f:
    json.dump(selected_sentences, f, indent=4)

with open(join(dirname(__file__), 'word_to_selected_sentence.json'), 'w') as f:
    json.dump(word_to_selected_sentence, f, indent=4)