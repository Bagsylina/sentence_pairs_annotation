from prodigy.components.db import connect
import pandas as pd
import ranking

#read database from sqllite
db = connect()
dataset = db.get_dataset_examples('sentence_simplest_test') #can be changed with any database name

dataframe = pd.DataFrame(dataset) #data is saved as a dataframe

#create a dictionary for each phrase in order to rank its' words
#create a list of matches for pairwise-ranking in order to be able to generate the ranks and scores
pairwise_ranking_data = {}
for i in range(dataframe.shape[0]):
    #verify if selection was not ignored and if an option was chosen
    if len(dataframe.loc[i]["accept"]) > 0 and dataframe.loc[i]["accept"][0] != "None":
        sentence_id = dataframe.loc[i]["id"]
        accepted_word = dataframe.loc[i]["accept"][0]

        #add the sentence to the dictionary if not already there
        if not (sentence_id in pairwise_ranking_data):
            pairwise_ranking_data[sentence_id] = {"matches": [], "ranks": [], "scores" : []}

        if accepted_word == dataframe.loc[i]["word_pair"][0]:
            pairwise_ranking_data[sentence_id]["matches"].append({"winner": dataframe.loc[i]["word_pair"][0], "loser": dataframe.loc[i]["word_pair"][1]})
        else:
            pairwise_ranking_data[sentence_id]["matches"].append({"winner": dataframe.loc[i]["word_pair"][1], "loser": dataframe.loc[i]["word_pair"][0]})

#generate rankings and scores
for sentence_id in pairwise_ranking_data:
    pairwise_ranking_data[sentence_id]["ranks"] = ranking.ranks(pairwise_ranking_data[sentence_id]["matches"])
    pairwise_ranking_data[sentence_id]["scores"] = ranking.scores(pairwise_ranking_data[sentence_id]["matches"])
    
for sentence_id in pairwise_ranking_data:
    print(pairwise_ranking_data[sentence_id]["ranks"])
    print(pairwise_ranking_data[sentence_id]["scores"])
