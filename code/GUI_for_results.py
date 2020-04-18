'''
    GUI for the COVID-19 chatbot.
    Enter the query and it returns the list of top 5 papers
        which can answer the query.
'''

from tkinter import *

root_path = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge/'
from sentence_transformers import SentenceTransformer
import scipy.spatial
from tkinter import ttk as ttk
import pickle as pkl
import pandas as pd

root = Tk()
root_path = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge/'

def load_model():
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')
    return embedder

def retrieve_input():
    query = []
    inputValue = str(textBox.get("1.0","end-1c"))
    query.append(inputValue)
    #textBox.delete('1.0', END)
    print(query)
    bert_results(query)

def bert_results(queries):
    df = pd.read_csv(root_path + "covid_sentences_full.csv", index_col=0)
    with open(root_path + 'sentences_list.pkl', 'rb') as f:
        df_sentences_list = pkl.load(f)
    f.close()
    corpus = df_sentences_list
    with open(root_path + "corpus_embeddings.pkl" , "rb") as file_:
        corpus_embeddings = pkl.load(file_)
    file_.close()
    embedder = load_model()
    query_embeddings = embedder.encode(queries, show_progress_bar=True)
    closest_n = 1
    print("\nTop 5 most similar sentences in corpus:")
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n=========================================================")
        print("==========================Query==============================")
        print("===",query,"=====")
        print("=========================================================")

        for idx, distance in results[0:closest_n]:
            score = "Score: %.4f" % (1-distance)
            outbox.insert(END, "Query: " + str(query) + "\n\n") 
            outbox.insert(END, score + '\n\n')
            outbox.insert(END, "Paragraph: " + str(corpus[idx].strip()) + "\n\n")
            row_dict = df.loc[df.index== corpus[idx]].to_dict()
            outbox.insert(END, "Paper ID: " + str(row_dict["paper_id"][corpus[idx]]) + "\n\n")
            outbox.insert(END, "Title: " + str(row_dict["title"][corpus[idx]]) + "\n\n")
            outbox.insert(END, "Abstract: " + str(row_dict["abstract"][corpus[idx]]) + "\n\n")

            print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
            print("Paragraph:   ", corpus[idx].strip(), "\n" )
            
            print("paper_id:  " , row_dict["paper_id"][corpus[idx]] , "\n")
            print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
            print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
            print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
            print("-------------------------------------------")


textBox = Text(root, height=2, width=100)
textBox.pack()
outbox = Text(root, height=20, width=100)
outbox.pack()
buttonCommit = ttk.Button(root, text="Ask", 
                    command=lambda: retrieve_input())
buttonCommit.pack()
mainloop()