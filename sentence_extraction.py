from os import listdir
from os.path import isfile, join, dirname
from itertools import islice
import json
import spacy

#dictionary for sentences containing each word (max 1000 sentences)
#word set for words with less than 1000 sentences
word_to_sentence = {}
word_set = set([])

#tokenizer to get all sentences and words
nlp = spacy.blank("ro")
nlp.add_pipe("sentencizer")

#read word_list.json that contains a python list of words
with open(join(dirname(__file__), 'word_list.json'), 'r') as f:
    word_list = json.load(f)

for word in word_list:
    if word not in word_to_sentence:
        word_to_sentence[word] = []
        word_set.add(word)

#mean length and standard deviation
#mean_word_count = int(statistics.mean(sentence_lengths))
#stdev_word_count = int(statistics.stdev(sentence_lengths))
mean_word_count = 27
stdev_word_count = 9
#minimum and maximum sentence length wanted 
min_word_count = mean_word_count - 2 * stdev_word_count
max_word_count = mean_word_count + 2 * stdev_word_count

#open folder with all text files of the data
sentence_data_dir = join(dirname(__file__), "sentence_data")

#list with all text files
text_files = [f for f in listdir(sentence_data_dir) if isfile(join(sentence_data_dir, f))]

#process maximum 20 lines at once
n = 20

#process each file
for text_file in text_files:
    filename = join(sentence_data_dir, text_file)
    with open(filename, 'rb') as f:
        #process 20 lines at once
        for n_lines in iter(lambda: tuple(islice(f, n)), ()):
            texte = [str(x, 'utf-8') for x in n_lines]
            for doc in nlp.pipe(texte):
                for sentence in list(doc.sents):
                    sentence_word_count = len(sentence)
                    #for each sentence check length and then for each word check if it appears in word set
                    if sentence_word_count >= min_word_count and sentence_word_count <= max_word_count:
                        word_remove = []
                        for token in sentence:
                            word = str(token)
                            if word in word_set:
                                word_to_sentence[word].append(str(sentence))
                                #if word has reached 1000 sentences remove from set to stop adding new sentences
                                if len(word_to_sentence[word]) == 1000:
                                    word_remove.append(word)

                        for word in word_remove:
                            word_set.remove(word)

#save data to a json file
with open(join(dirname(__file__), 'word_to_sentence.json'), 'w') as f:
    json.dump(word_to_sentence, f)