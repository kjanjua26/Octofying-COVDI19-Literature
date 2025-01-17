import pandas as pd
import pickle as pkl
import numpy as np
import tqdm, os, glob, time
from sentence_transformers import SentenceTransformer
import scipy.spatial
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

root_path = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge/'

def test_BERT(queries):
    '''
        This is a CLI based tester for BERT.
        Takes in a list of queries and computes n-closed points.
        The distance is computed based on cosine similarity.
        This is taken from:
            https://github.com/theamrzaki/COVID-19-BERT-ResearchPapers-Semantic-Search#data-links
    '''

    df = pd.read_csv(root_path + "covid_sentences_full.csv", index_col=0)
    with open(root_path + 'sentences_list.pkl', 'rb') as f:
        df_sentences_list = pkl.load(f)
    f.close()
    corpus = df_sentences_list
    with open(root_path + "corpus_embeddings.pkl" , "rb") as file_:
        corpus_embeddings = pkl.load(file_)
    file_.close()
    query_embeddings = embedder.encode(queries, show_progress_bar=True)
    closest_n = 1
    print("\nTop 1 most similar sentences in corpus:")
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n=========================================================")
        print("==========================Query==============================")
        print("===",query,"=====")
        print("=========================================================")

        for idx, distance in results[0:closest_n]:
            print("Score:   ", "(Score: %.4f)" % (1-distance) , "\n" )
            print("Paragraph:   ", corpus[idx].strip(), "\n" )
            row_dict = df.loc[df.index== corpus[idx]].to_dict()
            print("paper_id:  " , row_dict["paper_id"][corpus[idx]] , "\n")
            print("Title:  " , row_dict["title"][corpus[idx]] , "\n")
            print("Abstract:  " , row_dict["abstract"][corpus[idx]] , "\n")
            print("Abstract_Summary:  " , row_dict["abstract_summary"][corpus[idx]] , "\n")
            print("-------------------------------------------")

queries = ['What has been published about medical care?',
           'Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest',
           'Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually',
           'Resources to support skilled nursing facilities and long term care facilities.',
           'Mobilization of surge medical staff to address shortages in overwhelmed communities .',
           'Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies .']

test_BERT(queries)