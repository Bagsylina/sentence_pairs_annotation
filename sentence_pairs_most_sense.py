import prodigy
from prodigy.components.stream import get_stream

#define recipe
@prodigy.recipe(
    "sentence_pairs"
)

#define function corresponding to recipe
#needs to return a dictionary that defines the settings for the annotation server
def sentence_pairs(dataset: str, file_path):

    #get data that needs to be annotated
    stream = get_stream(file_path)
    stream = add_choices_to_stream(stream)
    
    return {
        "dataset": dataset,
        "view_id": "choice",
        "stream": stream,
        "config": {
            "feed_overlap": True,
            "buttons": ["accept", "ignore"]
        }
    }

#modifies data for prodigy to be able to know about between which options needs to be chosen
#the rest of the data is still present, just new data appended
def add_choices_to_stream(stream):
    for task in stream:
        #search word in sentence and highlight it
        word0 = task["new_sentence1"][0].find(task["word_pair"][0])
        word1 = task["new_sentence2"][0].find(task["word_pair"][1])
        span0 = [{"start": word0, "end": word0 + len(task["word_pair"][0]), "label": "WORD"}]
        span1 = [{"start": word1, "end": word1 + len(task["word_pair"][1]), "label": "WORD"}]
        #taken the word as id, sentence as the text
        options = [{"id": task["word_pair"][0], "text": task["new_sentence1"][0], "spans": span0}, {"id": task["word_pair"][1], "text": task["new_sentence2"][0], "spans": span1}]
        task["options"] = options
        task["text"] = "Care intreabare are sens?"
        task["sentence_id"] = task["id"]
        yield task
