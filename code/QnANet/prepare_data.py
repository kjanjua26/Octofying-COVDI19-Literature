'''
    In this code we prepare data for the BERT based QnA model.


    1. Convert the data to sentences
    2. Get BERT embeddings for each sentence
    3. Get question and work.
'''
import glob, re, os, json, tqdm, re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

stop_words = set(stopwords.words('english')) 
basepath = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge'

def draw_horizontal_lines(times):
    print("="*times)

def get_rid_of_stopwords(text):
    word_tokens = word_tokenize(text) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]     
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)

def get_sentences(text, article_id):
    pattern = r'\[.*?\]'
    text = re.sub(pattern, '', text)
    sentences = text.split(' . ')
    sentences = [x.split(' ') for x in sentences if len(x.split(' ')) > 5]
    sentences = [' '.join(i) for i in sentences]
    article_id_lst = [article_id for x in range(len(sentences))]
    return (sentences, article_id_lst)

def read_all_files():
    all_file = [x for x in glob.glob(os.path.join(basepath + "/*/*/*.json"))]
    article_id = 0
    total_sents, article_ids_total = [], []
    for paper in tqdm.tqdm(all_file):
        article_id += 1
        read_paper = json.load(open(paper))
        title = read_paper['metadata']['title']
        try:
            abstract = read_paper['abstract'][0]['text']
        except:
            abstract = " "
        paper_text = ""
        for text in read_paper['body_text']:
            paper_text += text['text'] + '\n\n'
        paper_text = paper_text + ' ' +  abstract
        clean_text = get_rid_of_stopwords(paper_text.lower())
        snts, articles_id_lst = get_sentences(clean_text, article_id)
        total_sents.append(snts)
        article_ids_total.append(articles_id_lst)
        flatten = lambda l: [item for sublist in l for item in sublist]
    return (flatten(total_sents), flatten(article_ids_total))

def caller():
    corpus, article_ids = read_all_files()
    df_sent = pd.DataFrame({'article_id': [], 'sentence': []})
    df_sent['article_id'] = article_ids
    df_sent['sentence'] = corpus
    print(df_sent.head(10))
    print(df_sent.tail(10))
    df_sent.to_csv("df_sents.csv")

if __name__ == "__main__":
    caller()