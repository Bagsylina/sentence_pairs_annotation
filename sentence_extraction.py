from os import listdir
from os.path import isfile, join, dirname
from itertools import islice
import json
import spacy
import re

#mean length and standard deviation
mean_word_count = 27
stdev_word_count = 9
#minimum and maximum sentence length wanted 
min_word_count = mean_word_count - 2 * stdev_word_count
max_word_count = mean_word_count + 2 * stdev_word_count

#special characters to be excluded
excluded_characters = set(['*', '@', '[', ']', '\\', '/', '<', '>', '=', '^', '_', '`', '{', '}', '|', '~'])
nr_excluded = len(excluded_characters)

#romanian diacritics and punctuation
romanian_diacritics = set([536, 537, 538, 539, 350, 351, 354, 355, 258, 259, 194, 226, 206, 238, 8220, 8211, 8212, 8221, 8222, 8230, 8217, 770, 807, 806, 774, 171, 187])

#ignore sentence if it checks certain criterias
def validate_sentence(sentence):
    #check sentence length based on mean and stdev
    sentence_token_count = len(sentence)
    if sentence_token_count < min_word_count or sentence_token_count > max_word_count:
        return False
    
    #get sentence and all its' words in string format
    sentence_tokens = [str(x) for x in sentence]
    sent_str = str(sentence)

    #check if sentence contains special characters
    sentence_chars = set(sent_str)
    if len(excluded_characters - sentence_chars) < nr_excluded:
        return False
    
    #check if more then 50% of all characters are uppercase
    nr_upper = sum(1 for c in sent_str if c.isupper())
    if nr_upper >= len(sent_str) / 2:
        return False
    
    #check if more than 50% of tokens contain digits
    nr_with_digits = sum(1 for t in sentence_tokens if bool(re.search(r'\d', t)))
    if nr_with_digits >= sentence_token_count / 2:
        return False
    
    #check if sentence contains more than one token longer than 20 characters
    nr_long = sum(1 for t in sentence_tokens if len(t) > 20)
    if nr_long >= 1:
        return False
    
    #check if more than 60% of tokens are capitalized or numbers (are names, trademarks, dates etc.)
    nr_capital = sum(1 for t in sentence_tokens if t[0].isupper() or t[0].isdigit())
    if nr_capital >= sentence_token_count * 3 / 5:
        return False
    
    #check if sentence contains non-ascii characters (excluding diacritics and romanian punctuation)
    nr_non_ascii = sum(1 for c in sent_str if ord(c) > 127 and ord(c) not in romanian_diacritics)
    if nr_non_ascii >= 1:
        return False
    
    #check if sentence contains more than 4 consecutive punctuation characters
    charsearch = re.search("[.?!,;:'\u201c\u201d\u201b\u2013\u2014\u2019\u2025]{5,}", sent_str)
    if bool(charsearch):
        return False
    
    #sentence passes all checks
    return True

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
                    #for each sentence check if it's valid and then for each word check if it appears in word set
                    if validate_sentence(sentence):
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