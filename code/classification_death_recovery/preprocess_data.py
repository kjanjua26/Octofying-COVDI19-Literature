'''
    This code file is to preprocess the pandas .csv file made
    and return the arrays required for training.
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

basepath = '/Users/Janjua/Desktop/Projects/TAWork/COVID19Testing/'

def flatten(L):
    '''
        Flatten a multi-dimensional list of lists.
        Input: List (L) to flatten
        Output: Yields the flatten list
    '''
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

def clean_Y_vector(Y_1, Y_2):
    '''
        A short snippet to clean the Y vector.
        It has some value issues.
        Input: The .csv death and recovered columns (Y_1 and Y_2)
        Returns: Arrays Y1 and Y2
    '''
    Y1, Y2 = [], []
    for i in Y_1:
        if i == '0':
            Y1.append(0)
        elif i == '1':
            Y1.append(1)
        else:
            Y1.append(1)
    for i in Y_2:
        if i == '0':
            Y2.append(0)
        elif i == '1':
            Y2.append(1)
        else:
            Y2.append(1)
    Y1 = to_categorical(Y1, num_classes=2)
    Y2 = to_categorical(Y2, num_classes=2)
    Y1 = np.array(Y1).astype(float)
    Y2 = np.array(Y2).astype(float)
    return (Y1, Y2)

def read_prep_data():
    '''
        Read the data from the .csv file and prepare it.
        Converts the features to list and returns the list.
    '''
    data = pd.read_csv(basepath + 'mod_fin.csv')
    temp_X = data[['reporting date', 'summary', 'location', 'country', 'gender', 'age', 'symptom', 
             'visiting Wuhan', 'from Wuhan']]
    temp_X["age"].fillna(45, inplace = True)
    temp_X["reporting date"].fillna(0, inplace = True)
    temp_X["from Wuhan"].fillna(1, inplace = True)
    temp_X["visiting Wuhan"].fillna(1, inplace = True)

    Y_1 = data['death']
    Y_2 = data['recovered']

    X = []
    print("Preparing the X vector!")
    for ix, row in tqdm(temp_X.iterrows()):
        local_holder = []
        summary_arr = [float(x) for x in row['summary'][1:-1].split()]
        symptom_arr = [float(x) for x in row['symptom'][1:-1].split()]
        loc_arr = [float(x) for x in row['location'][1:-1].split()]
        country_arr = [float(x) for x in row['country'][1:-1].split()]
        gender_arr = [float(x) for x in row['gender'][1:-1].split()]
        
        local_holder.append(float(row['reporting date']))
        local_holder.append(summary_arr)
        local_holder.append(loc_arr)
        local_holder.append(country_arr)
        local_holder.append(gender_arr)
        local_holder.append(float(row['age']))
        local_holder.append(symptom_arr)
        local_holder.append(float(row['visiting Wuhan']))
        local_holder.append(float(row['from Wuhan']))
        X.append(list(flatten(local_holder)))
    X = np.asarray(X)
    Y1, Y2 = clean_Y_vector(Y_1, Y_2)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X, Y1)
    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X, Y2)
    return (X_train1, X_test1, Y_train1, Y_test1, 
            X_train2, X_test2, Y_train2, Y_test2)