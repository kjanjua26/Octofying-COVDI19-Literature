'''
    This code is used to train a few layer fully connected 
    network to classify given a set of features, if a person is dead or alive.
    Since we want to learn whether given X, Y1 (death or no death)
    and X, Y2 (recovered or no recovered) so we train in two settings.
'''
import preprocess_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization

def net():
    '''
        The basic FC-7 neural architecture
    '''
    model = Sequential()
    model.add(BatchNormalization())
    model.add(Dense(32, activation = "sigmoid"))
    model.add(Dense(64, activation = "tanh"))
    model.add(Dense(128, activation = "relu"))
    model.add(Dense(512, activation = "sigmoid"))
    model.add(Dense(512, activation = "tanh"))
    model.add(Dense(512, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(2, activation = "softmax"))
    return model

def compile_optimize():
    '''
        Compile the model and optimize it.
    '''
    model = net()
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

def train_net():
    '''
        Pass the data and train the neural network.
        Train for both cases, test for both cases.
    '''
    X_train1, X_test1, Y_train1, Y_test1, 
        X_train2, X_test2, 
        Y_train2, Y_test2 = preprocess_data.read_prep_data()
    print("Read data!")
    print("X_train1, X_train2: ", X_train1.shape, X_train2.shape)
    print("Y_train1, Y_train2: ", Y_train1.shape, Y_train2.shape)
    print("X_test1, X_test2: ", X_test1.shape, X_test2.shape)
    print("Y_test1, Y_test2: ", Y_test1.shape, Y_test2.shape)
    
    print("Training for first case: Given X, predict if death or no death!")
    model.fit(X_train1, Y_train1, epochs = 30, batch_size = 16)
    predictions = model.predict(X_test1)
    score = model.evaluate(X_test1, Y_test1)
    print('Test Accuracy in Case # 01: ', score[1])

    print("Training for second case: Given X, predict if recovered or no recovered!")
    model.fit(X_train1, Y_train2, epochs = 100, batch_size = 16)
    predictions = model.predict(X_test2)
    score = model.evaluate(X_test2, Y_test2)
    print('Test Accuracy in Case # 02: ', score[1])

if __name__ == "__main__":
    train_net()