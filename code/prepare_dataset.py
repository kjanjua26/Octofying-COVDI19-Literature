import pandas as pd
import tqdm, os, glob, json, re, time
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

BASEPATH = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/dataset'
stop_words = set(stopwords.words('english')) 

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

def load_data_for_training_w2v(isShort=False, till=10):
    """
        Loads the data for training and testing for the word2vec model.
    """
    data = pd.read_csv(BASEPATH + '/COVID_19_Lit.csv')
    corpus = data.drop(["paper_id", "abstract", "abstract_word_count", "body_text_word_count", "authors", "title", "journal"], axis=1)
    print(corpus.head(1))
    words, n_gram_total = [], []
    print(len(corpus))
    start = time.time()
    # For quick testing, we take a shorter subset of the data
    if isShort:
        for ix in range(0, len(corpus[:till])):
            words.append(str(corpus.iloc[ix]['body_text'][1:-1]).split(" "))
    else:
        for ix in range(0, len(corpus)):
            words.append(str(corpus.iloc[ix]['body_text'][1:-1]).split(" "))
    for word in words:
        n_gram = []
        for i in range(len(word)-2+1):
            n_gram.append("".join(word[i:i+2]))
        n_gram_total.append(n_gram)
    end = time.time()
    print("Prepared n-grams in: {}s".format(end-start))
    word2int, int2word = {}, {}
    start = time.time()
    for i, word in enumerate(n_gram_total[0]):
        word2int[word] = i
        int2word[i] = word
    word_with_neighbor = list(map(list, zip(n_gram_total[0], n_gram_total[0][1:])))
    end = time.time()
    print("Computed neighbours in: {}s".format(end-start))
    X, y = [], []
    vocab_size = max(word2int.values()) + 1
    print("Vocab size: ", vocab_size)
    start = time.time()
    for word_neigh in word_with_neighbor:
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