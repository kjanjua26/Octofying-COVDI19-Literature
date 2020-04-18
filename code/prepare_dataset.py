import pandas as pd
import tqdm, os, glob, json, re, time
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import enchant
import pickle as pkl

BASEPATH = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset'
stop_words = set(stopwords.words('english')) 
engDict = enchant.Dict("en_US")
root_path = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset/CORD-19-research-challenge/'

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
    data = pd.read_csv(BASEPATH + "/CORD-19-research-challenge/metadata.csv")
    json_files = glob.glob(BASEPATH + "/CORD-19-research-challenge/*/*/*.json", recursive=True)
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
    # Removing preposition marks
    covid_df['body_text'] = covid_df['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','', x))
    covid_df['abstract'] = covid_df['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','', x))
    # Convert to lower case
    covid_df['body_text'] = covid_df['body_text'].apply(lambda x: x.lower())
    covid_df['abstract'] = covid_df['abstract'].apply(lambda x: x.lower())

    covid_df.to_csv(BASEPATH + '/COVID_19_Lit.csv', encoding='utf-8', index=False)
    print(covid_df.head())
    print("Written dataframe to .csv file.")

def to_one_hot(data_point_index, vocab_size):
    """
        Converts numbers to one hot vectors
        # Returns: a one hot vector temp
        # Credits: 
            Function taken from:
                https://gist.github.com/aneesh-joshi/c8a451502958fa367d84bf038081ee4b
    """
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

def load_data_for_training_w2v():
    """
        Loads the data for training and testing for the word2vec model.
    """
    data = pd.read_csv(BASEPATH + '/COVID_19_Lit.csv')
    corpus = data.drop(["paper_id", "abstract", "abstract_word_count", "body_text_word_count", "authors", "title", "journal"], axis=1)
    print(corpus.head(1))
    words, n_gram = [], []
    print(len(corpus))

    start = time.time()
    for ix in range(0, len(corpus)):
        words.append(str(corpus.iloc[ix]['body_text'][1:-1]).split(" "))
    print('Word Length: ', len(words))
    for word in words:
        for i in range(len(word)-2+1):
            word1, word2 = word[i:i+2]
            if word1 != "" and word2 != "":
                if engDict.check(word1) == True and engDict.check(word2) == True:
                    n_gram.append("".join(word[i:i+2]))
    end = time.time()

    print("Prepared n-grams in: {}s".format(end-start))
    print("N-gram length: ", len(n_gram))
    n_gram = n_gram[:100000]
    print("Reducing size to: ", len(n_gram))
    word2int, int2word = {}, {}
    print("N-gram length: ", len(n_gram))
    start = time.time()

    for i, word in enumerate(n_gram):
        word2int[word] = i
        int2word[i] = word
    word_with_neighbor = list(map(list, zip(n_gram, n_gram[1:])))
    end = time.time()
    print("Computed neighbours in: {}s".format(end-start))

    X, y = [], []
    vocab_size = max(word2int.values()) + 1
    print("Vocab size: ", vocab_size)
    start = time.time()
    for idx, word_neigh in enumerate(word_with_neighbor):
        if idx % (len(word_with_neighbor) // 10) == 0:
            print('Processing: {} of {}'.format(idx, len(word_with_neighbor)))
        X.append(to_one_hot(word2int[word_neigh[0]], vocab_size))
        y.append(to_one_hot(word2int[word_neigh[1]], vocab_size))
    X = np.asarray(X)
    y = np.asarray(y)
    end = time.time()
    print("Prepared the data vectors: {}s".format(end-start))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    print("Shapes: \nX_train: {}\ny_train: {}\nX_test: {}\ny_test: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    np.save('arrays/X_train_w2v.npy', X_train)
    np.save('arrays/y_train_w2v.npy', y_train)
    np.save('arrays/X_test_w2v.npy', X_test)
    np.save('arrays/y_test_w2v.npy', y_test)
    print("Saved arrays!")

def read_arrays_and_return():
    """
        Reads the prepared numpy arrays
        # Returns: the read np arrays
    """
    X_train = np.load('arrays/X_train_w2v.npy')
    y_train = np.load('arrays/y_train_w2v.npy')
    X_test = np.load('arrays/X_test_w2v.npy')
    y_test = np.load('arrays/y_test_w2v.npy')
    return (X_train, X_test, y_train, y_test)

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            try:
              self.paper_id = content['paper_id']
            except:
              pass
              #self.paper_id = str(content['paper_id'])
            self.abstract = []
            self.body_text = []
            # Abstract
            try:
              for entry in content['abstract']:
                  self.abstract.append(entry['text'])
            except:
              pass
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data

def prepare_dataset_for_BERT():
    print("Preparing dataset for BERT!")
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    print('Len of all files: ', len(all_json))
    metadata_path = f'{root_path}/metadata.csv'
    meta_df = pd.read_csv(metadata_path, dtype={
        'pubmed_id': str,
        'Microsoft Academic Paper ID': str, 
        'doi': str
    })
    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}

    for idx, entry in enumerate(all_json):
        if idx % (len(all_json) // 10) == 0:
            print(f'Processing index: {idx} of {len(all_json)}')
        content = FileReader(entry)

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        if len(meta_data) == 0:
            continue
        dict_['paper_id'].append(content.paper_id)
        dict_['abstract'].append(content.abstract)
        dict_['body_text'].append(content.body_text)    
        if len(content.abstract) == 0: 
            dict_['abstract_summary'].append("Not provided.")
        elif len(content.abstract.split(' ')) > 100:
            info = content.abstract.split(' ')[:100]
            summary = get_breaks(' '.join(info), 40)
            dict_['abstract_summary'].append(summary + "...")
        else:
            summary = get_breaks(content.abstract, 40)
            dict_['abstract_summary'].append(summary)
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        try:
            authors = meta_data['authors'].values[0].split(';')
            if len(authors) > 2:
                dict_['authors'].append(". ".join(authors[:2]) + "...")
            else:
                dict_['authors'].append(". ".join(authors))
        except Exception as e:
            dict_['authors'].append(meta_data['authors'].values[0])
        try:
            title = get_breaks(meta_data['title'].values[0], 40)
            dict_['title'].append(title)
        except Exception as e:
            dict_['title'].append(meta_data['title'].values[0])
        dict_['journal'].append(meta_data['journal'].values[0])
    
    print('Paper ID len: ', len(dict_['paper_id']))
    print('Abstract len: ', len(dict_['abstract']))
    print('Text len: ', len(dict_['body_text']))
    print('Title len: ', len(dict_['title']))
    print('Journal len: ', len(dict_['journal']))
    print('Abstract Summary len: ', len(dict_['abstract_summary']))
    
    df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
    print("Data Cleaning!")
    
    df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
    df_covid.dropna(inplace=True)
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
    df_covid['body_text'] = df_covid['body_text'].apply(lambda x: x.lower())
    df_covid['abstract'] = df_covid['abstract'].apply(lambda x: x.lower())
    df_covid.to_csv(root_path + "covid.csv")

def preprocess_for_BERT():
    if os.path.exists(root_path + "covid.csv"):
        df_covid_test = pd.read_csv(root_path + "covid.csv")
        text = df_covid_test.drop(["authors", "journal", "Unnamed: 0"], axis=1)
        text_dict = text.to_dict()
        len_text = len(text_dict["paper_id"])
        print('Text Len: ', len_text)
        paper_id_list, body_text_list = [], []
        title_list, abstract_list, abstract_summary_list = [], [], []

        for i in tqdm.tqdm(range(0, len_text)):
            paper_id = text_dict["paper_id"][i]
            body_text = text_dict["body_text"][i].split("\n")
            title = text_dict["title"][i]
            abstract = text_dict["abstract"][i]
            abstract_summary = text_dict["abstract_summary"][i]
            for b in body_text:
                paper_id_list.append(paper_id)
                body_text_list.append(b)
                title_list.append(title)
                abstract_list.append(abstract)
                abstract_summary_list.append(abstract_summary)

        print('Writing initial sentences to CSV file!')
        df_sentences = pd.DataFrame({"paper_id": paper_id_list}, index=body_text_list)
        df_sentences.to_csv(root_path + "covid_sentences.csv")

        print('Writing complete sentences to CSV file!')
        df_sentences = pd.DataFrame({"paper_id":paper_id_list,"title":title_list,"abstract":abstract_list,"abstract_summary":abstract_summary_list},index=body_text_list)
        df_sentences.to_csv(root_path + "covid_sentences_full.csv")
    else:
        print("Call function: prepare_dataset_for_BERT() first.")
        print("The file is not present!")

def finalize_data_for_BERT():
    if os.path.exists(root_path + "covid_sentences.csv") and \
        os.path.exists(root_path + "covid_sentences_full.csv"):
        df_sentences = pd.read_csv(root_path + "covid_sentences.csv")
        df_sentences = df_sentences.set_index("Unnamed: 0")
        df_sentences = df_sentences["paper_id"].to_dict()
        df_sentences_list = list(df_sentences.keys())
        print('Sentence List Len: ', len(df_sentences_list))
        df_sentences_list = [str(d) for d in tqdm.tqdm(df_sentences_list)]
        print("Len of sentence_lst: ", len(df_sentences_list))
        print("Pickling the data!")
        with open(root_path + 'sentences_list.pkl', 'wb') as f:
            pkl.dump(df_sentences_list, f)
        print("Done serialization!")
    else:
        print("Call function: preprocess_for_BERT() first.")
        print("The file is not present!")
            
finalize_data_for_BERT()