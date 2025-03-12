from nltk import agreement
import json
from os.path import join, dirname
import numpy as np
from collections import defaultdict
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from copy import copy
from wordfreq import zipf_frequency
import pandas as pd
from scipy.stats import pearsonr
scaler = MinMaxScaler()



def map_label(label):
    match label:
        case "foarte familiar":
            return 1
        case "simplu":
            return 2
        case "nici simplu nici complex":
            return 3
        case "complex":
            return 4
        case "nu cunosc":
            return 5
        case _:
            return -1

def map_label0(label):
    match label:
        case "foarte familiar":
            return 1
        case "simplu":
            return 1
        case "nici simplu nici complex":
            return 2
        case "complex":
            return 3
        case "nu cunosc":
            return 3
        

def clean(text):
    text = ' '.join(text.split())
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" >.", ".")
    return text


def get_key(data):
    return (clean(data['text']), data['meta']['word'])


#filenames = ['victor.jsonl', 'sergiu.jsonl', 'stadio88.jsonl', 'petru.jsonl', 'iulia.jsonl', 'user.jsonl']
filenames = ['iulia.jsonl', 'user.jsonl']
filenames = ['victor.jsonl', 'sergiu.jsonl', 'stadio88.jsonl', 'petru.jsonl']
filenames = ['stadio88.jsonl', 'sergiu.jsonl']
filenames = ['victor.jsonl', 'stadio88.jsonl', 'sergiu.jsonl']
filenames = ['victor.jsonl', 'sergiu.jsonl', 'stadio88.jsonl', 'petru.jsonl', 'user.jsonl']
filenames = ['victor.jsonl', 'stadio88.jsonl', 'petru.jsonl', 'user.jsonl']
# best ->
filenames = ['victor.jsonl', 'stadio88.jsonl',  'user.jsonl']
# second best ->
filenames = ['victor.jsonl', 'stadio88.jsonl', 'user.jsonl', 'petru.jsonl']
filenames = ['victor.jsonl', 'stadio88.jsonl', 'user.jsonl', 'petru.jsonl', 'sergiu.jsonl']
#filenames = ['victor.jsonl', 'stadio88.jsonl', 'user.jsonl', 'petru.jsonl',  'sergiu.jsonl', 'iulia.jsonl']


annotators = [fis.replace('.jsonl', '') for fis in filenames]


data_files = []
for filename in filenames:
    with open(join(dirname(__file__), filename), 'r', encoding="UTF-8") as f:
        lines = f.readlines()
    all_lines = []
    seen_already = set()
    for line in lines:
        js = json.loads(line)
        js['text'] = clean(js['text'])
        key = get_key(js)
        if key in seen_already:
            continue
        seen_already.add(key)
        all_lines.append(js)
    data_files.append(all_lines)

key2id = {}
id2key = {}
idx = 0
for line in data_files[0]:
    key = get_key(line)
    if key not in key2id:
        key2id[key] = idx
        id2key[idx] = key
        idx += 1

def get_id(data):
    key = get_key(data)
    #return key2id[key]
    return key

elements = defaultdict(list)
data = []
for annotator, data_list in zip(annotators, data_files):
    for elem in data_list:
        key = get_id(elem)
        elem['text'] = clean(elem['text'])
        elem["annotator"] = annotator
        if len(elements[key]) == len(annotators):
            print('Duplicate:', key)
            continue
        elements[key].append(elem)
        data.append(elem)


taskdata = []
for k, v in elements.items():
    for elem in v:
        taskdata.append([elem['annotator'], k, map_label(elem['label'][0][2])])


task = agreement.AnnotationTask(data=sorted(taskdata, key=lambda x: x[0]))
print(1)
iaa_results = {
    "n_examples": len(taskdata),
    "n_categories": len(set([x[2] for x in taskdata])),
    "n_annotators": len(annotators),
    "percent_agreement": np.round(task.avg_Ao(), 3),
    "cohen_kappa": np.round(task.kappa(), 3),
    "kripp_alpha": np.round(task.alpha(), 3)
}
pprint(iaa_results)

with open(join(dirname(__file__), 'labeling_iaa_results.json'), 'w') as f:
    json.dump(iaa_results, f, indent=4)

scaler.fit(np.array(list(set([x[2] for x in taskdata]))).reshape(-1, 1)) 
data_dict = {}
for id, elems in elements.items():
    if len(elems) != len(annotators):
        print('Missing:', id)
        continue
    data_dict[id] = copy(elems[0])
    del data_dict[id]['annotator']
    del data_dict[id]['label']
    data_dict[id]['labels'] = [elem['label'][0] + [elem['annotator']] for elem in elems]
    scores = [map_label(elem['label'][0][2]) for elem in elems]
    data_dict[id]['scores'] = scaler.transform(np.array(scores).reshape(-1, 1)).reshape(1,-1)[0].tolist()
    data_dict[id]['mean_score'] = np.average(data_dict[id]['scores'])
    data_dict[id]['variance'] = np.var(data_dict[id]['scores'])
    

data_list = sorted(data_dict.values(), reverse=True, key=lambda x: x['variance'])

with open(join(dirname(__file__), 'labeling_scores.jsonl'), 'w') as f:
    for entry in data_list:
        json.dump(entry, f)
        f.write('\n')



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
idx = df.frq>0.00001
df2 = df[idx]


print(pearsonr(df.frq, df['mean']))
iaa_updated = {
    "correlation_with_zipf": np.round(pearsonr(df.frq, df['mean'])[0], 2),
    "correlation_with_zipf_filtered": np.round(pearsonr(df2.frq, df2['mean'])[0], 2)
}
iaa_results.update(iaa_updated)
iaa_results['counts'] = json.loads(df['majority'].value_counts().to_json())
pprint(iaa_results)

with open(join(dirname(__file__), 'labeling_iaa_results.json'), 'w') as f:
    json.dump(iaa_results, f, indent=4)