'''
    
'''
import datetime, os
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# defining the data path
basepath = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/code/classification_death_recovery/'
datapath = 'updated_on_2020-04-27'

def update_dataset():
    tday = str(datetime.date.today())
    data = "updated_on_{}".format(tday)
    toCreate = basepath + data
    unzip_the_files = "unzip {} -d {}".format(basepath + "novel-corona-virus-2019-dataset.zip", toCreate)
    print(unzip_the_files)
    os.makedirs(toCreate)
    os.system('kaggle datasets download sudalairajkumar/novel-corona-virus-2019-dataset')
    error = os.system(unzip_the_files)
    if not error:
        print("Updated the dataset!")

def map_dates_to_ints(date_val):
    split_date = date_val.split('/')
    initial = float(split_date[0])
    final = float(split_date[1])
    return ((initial * final ** 2) ** 0.5)

def clean_words(word):
    word = word.lower()
    if ',' in word:
        try:
            required_word, _ = word.split(',')
            return required_word
        except:
            return word
    elif ' ' in word:
        return ''.join(word.split(' '))
    else:
        return word

def d2v_for_summary_symptoms(tagged_data):
    print("Training D2V!")
    max_epochs = 10
    vec_size = 100
    alpha = 0.025
    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(max_epochs):
        print("Epoch: {}, Alpha: {}".format(epoch, model.alpha))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha
    model.save(basepath + "d2v.model")
    print("Model Saved")

def get_embeddings_d2v(sent):
    model = Doc2Vec.load(basepath + "d2v.model")
    to_compute = word_tokenize(str(sent).lower())
    v1 = model.infer_vector(to_compute)
    return v1

def apply_embedding(data):
    data['summary'] = data['summary'].apply(get_embeddings_d2v)
    data['symptom'] = data['symptom'].apply(get_embeddings_d2v)
    data['location'] = data['location'].apply(get_embeddings_d2v)
    data['country'] = data['country'].apply(get_embeddings_d2v)
    data['gender'] = data['gender'].apply(get_embeddings_d2v)
    data.to_csv('mod_fin.csv')
    print("Done Embeddifying and saved .CSV!")

def caller():
    # reading the data
    data = pd.read_csv(os.path.join(basepath, datapath) + '/COVID19_line_list_data.csv')
    data = data[['reporting date', 'summary', 'location', 'country', 'gender', 'age', 'symptom', 
                'visiting Wuhan', 'from Wuhan', 'death', 'recovered']]

    data['reporting date'].fillna("1/21/2020", inplace=True)
    data['reporting date'] = data['reporting date'].apply(map_dates_to_ints)
    data['location'] = data['location'].apply(clean_words)
    data['country'] = data['country'].apply(clean_words)

    summary = data['summary'].tolist()
    symptoms = data['symptom'].tolist()
    location = data['location'].tolist()
    countries = data['country'].tolist()
    gender = data['gender'].tolist()
    vocab = summary + symptoms + location + countries + gender
    tagged_data_d2v = [TaggedDocument(words=word_tokenize(str(_d).lower()), 
                                        tags=[str(i)]) for i, _d in enumerate(vocab)]
    d2v_for_summary_symptoms(tagged_data_d2v)
    print("Embeddings made, applying now!")
    apply_embedding(data)

if __name__ == "__main__":
    update_dataset()
    caller()