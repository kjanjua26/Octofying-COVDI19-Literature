import pandas as pd
import tqdm, os, glob, json
import numpy as np

BASEPATH = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge'

def retrieve_data_from_json(file_path):
    """
        Reads the json file and returns the necessary items.
        # Arguments: 
            file_path: the path to the .json file
    """
    with open(file_path) as file:
        data = json.loads(file.read())
        abstract, full_text = [], []
        abstract = str([x['text'] for x in data['abstract']])
        full_text = str([x['text'] for x in data['body_text']])
        paper_id = data['paper_id']
        return (paper_id, abstract, full_text)

def prepare_dataset():
    """
        Reads the downloaded .csv file and performs some pre-processing on the data.
        # Returns: A dataframe file is returned which has cleaned data columns
                by removing the un-necessary information from the previous csv file.
        # Credits:
            Some aspects of code borrowed from: 
                https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool
    """
    data = pd.read_csv(BASEPATH + "/metadata.csv")
    json_files = glob.glob(BASEPATH + "/*/*/*.json", recursive=True)
    covid_data_dict = {'paper_id': [],
                        'abstract': [],
                        'body_text': [],
                        'authors': [],
                        'title': [],
                        'journal': []}
    for idx, entry in enumerate(json_files):
        if idx % (len(json_files) // 10) == 0:
            print('Processing: {} of {}'.format(idx, len(json_files)))
        paper_id, abstract, full_text = retrieve_data_from_json(entry)
        meta = data.loc[data['sha'] == paper_id]
        if len(meta) == 0:
            continue
        covid_data_dict['paper_id'].append(paper_id)
        covid_data_dict['abstract'].append(abstract)
        covid_data_dict['body_text'].append(full_text)
        try:
            authors = meta['authors'].values[0].split(';')
            if len(authors) > 2:
                covid_data_dict['authors'].append(authors[:1] + "...")
            else:
                covid_data_dict['authors'].append(". ".join(authors))
        except:
            covid_data_dict['authors'].append(". ".join(authors))      
        covid_data_dict['title'].append(meta['title'].values[0])
        covid_data_dict['journal'].append(meta['journal'].values[0])
    covid_df = pd.DataFrame(covid_data_dict, columns=['paper_id', 'abstract', 'body_text', \
                                                    'authors', 'title', 'journal'])
    covid_df['abstract_word_count'] = covid_df['abstract'].apply(lambda x: len(x.strip().split()))
    covid_df['body_text_word_count'] = covid_df['body_text'].apply(lambda x: len(x.strip().split()))

    covid_df.to_csv('COVID_19_Lit.csv', encoding='utf-8', index=False)
    print(covid_df.head())
    print("Written dataframe to .csv file.")

prepare_dataset()