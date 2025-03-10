from nltk import agreement
import json
from os.path import join, dirname
import numpy as np

with open(join(dirname(__file__), 'victor.jsonl'), 'r', encoding="UTF-8") as f:
    data1 = [json.loads(line) for line in f]

with open(join(dirname(__file__), 'sergiu.jsonl'), 'r', encoding="UTF-8") as f:
    data2 = [json.loads(line) for line in f]

with open(join(dirname(__file__), 'petru.jsonl'), 'r', encoding="UTF-8") as f:
    data3 = [json.loads(line) for line in f]

with open(join(dirname(__file__), 'stadio88.jsonl'), 'r', encoding="UTF-8") as f:
    data4 = [json.loads(line) for line in f]

taskdata = []

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
            return 5
        
new_ids = {}

data = []

for x in data1:
    y = x
    y["annotator"] = "victor"
    data.append(y)
for x in data2:
    y = x
    y["annotator"] = "sergiu"
    data.append(y)
for x in data3:
    y = x
    y["annotator"] = "petru"
    data.append(y)
for x in data4:
    y = x
    y["annotator"] = "stadio88"
    data.append(y)

for x in data:
    key = x['text'] + x['meta']['word']
    if x['id'] < 5000:
        x['text'] = ' '.join(x['text'].split())
        x['text'] = x['text'].replace(" ,", ",")
        x['text'] = x['text'].replace(" .", ".")
        x['text'] = x['text'].replace(" >.", ".")
        key = x['text'] + x['meta']['word']

    if key not in new_ids:
        new_ids[key] = x['id']
    else:
        x['id'] = new_ids[key]

for x in data1:
    taskdata.append(['victor', x['id'], map_label(x['label'][0][2])])
for x in data2:
    taskdata.append(['sergiu', x['id'], map_label(x['label'][0][2])])
for x in data3:
    taskdata.append(['petru', x['id'], map_label(x['label'][0][2])])
for x in data4:
    taskdata.append(['stadio88', x['id'], map_label(x['label'][0][2])])

task = agreement.AnnotationTask(data=taskdata)
iaa_results = {"n_examples": len(taskdata), "n_categories": 5, "n_annotators": 4, "percent_agreement": task.avg_Ao(), "cohen_kappa": task.kappa(), "kripp_alpha": task.alpha()}

with open(join(dirname(__file__), 'labeling_iaa_results.json'), 'w') as f:
    json.dump(iaa_results, f, indent=4)

data = data1 + data2 + data3 + data4
data_dict = {}

for x in data:
    id = x['id']

    if id not in data_dict:
        data_dict[id] = {'id': id, 'text': x['text'], 'meta': x['meta'], 'labels': [], 'scores': []}
    
    data_dict[id]['labels'].append(x['label'][0] + [x['annotator']])

    score_text = x['label'][0][2]
    score = (map_label(score_text) - 1) / 4
    data_dict[id]['scores'].append(score)

for id in data_dict:
    data_dict[id]['mean_score'] = np.average(data_dict[id]['scores'])
    data_dict[id]['variance'] = np.var(data_dict[id]['scores'])     

data_list = [data_dict[id] for id in data_dict]
data_list.sort(reverse=True, key=lambda x: x['variance'])

with open(join(dirname(__file__), 'labeling_scores.jsonl'), 'w') as f:
    for entry in data_list:
        json.dump(entry, f)
        f.write('\n')