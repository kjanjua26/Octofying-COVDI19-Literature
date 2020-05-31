'''
    This is the inference code for the model.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import preprocess_data
from train_pytorch import Net

def load_model_saved(model_file):
    '''
        Load the model and set it in inference mode.
    '''
    model = torch.load(model_file)
    model.eval()
    print(model)
    return model

def validate_compute_accuracy(model_file, X_test1, Y_test1):
    '''
        Compute the accuracy scores given X and y.
    '''
    net = load_model_saved(model_file)
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_test1)):
            real_class = torch.argmax(Y_test1[i])
            net_out = net(X_test1[i].view(-1, 504))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1
    return round(correct/total, 3)

def tester(model_file, load=False):
    if load:
        # load the files
        X_test_1 = np.load('x_test_1.npy')
        X_test_2 = np.load('X_test_2.npy')
        y_test_1 = np.load('y_test_1.npy')
        y_test_2 = np.load('y_test_2.npy')
    else:
        X_train1, X_test1, Y_train1, Y_test1, \
            X_train2, X_test2, \
            Y_train2, Y_test2 = preprocess_data.read_prep_data()
        X_train1 = torch.from_numpy(X_train1).float()
        X_train2 = torch.from_numpy(X_train2).float()
        X_test1 = torch.from_numpy(X_test1).float()
        X_test2 = torch.from_numpy(X_test2).float()
        
        Y_train1 = torch.from_numpy(Y_train1).float()
        Y_test1 = torch.from_numpy(Y_test1).float()
        Y_train2 = torch.from_numpy(Y_train2).float()
        Y_test2 = torch.from_numpy(Y_test2).float()

    print("Test Accuracy Case # 01: ", validate_compute_accuracy(model_file, X_test1, Y_test1))
    print("Test Accuracy Case # 02: ", validate_compute_accuracy(model_file, X_test2, Y_test2))


if __name__ == "__main__":
    model_file = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/code/classification_death_recovery/models/model_v1'
    tester(model_file)