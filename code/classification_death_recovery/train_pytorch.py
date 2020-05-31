'''
    This code is used to train a few layer fully connected 
    network to classify given a set of features, if a person is dead or alive.
    Since we want to learn whether given X, Y1 (death or no death)
    and X, Y2 (recovered or no recovered) so we train in two settings.
    Same as train.py, but in pytorch
    Code inspiration: 
    https://pythonprogramming.net/introduction-deep-learning-neural-network-pytorch/
'''

import preprocess_data
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

PATH = '/Users/Janjua/Desktop/Projects/Octofying-COVID19-Literature/code/classification_death_recovery/models/model_v2'
class Net(nn.Module):
    '''
        The basic FC-7 neural architecture class.
    '''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(504, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return F.log_softmax(x, dim=1)

def train_net(net, X_train1, Y_train1):
    '''
        Train the neural network using the MSELoss function and
            Adam optimizer.
    '''
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    BATCH_SIZE = 16
    loss_lst = []
    acc_lst = []
    for epoch in range(5):
        for i in range(0, len(X_train1), BATCH_SIZE):
            batch_X = X_train1[i:i+BATCH_SIZE]
            batch_y = Y_train1[i:i+BATCH_SIZE]
            net.zero_grad()
            outputs = net(batch_X.view(-1, 504))
            loss = loss_function(outputs, batch_y)
            acc = validate_compute_accuracy(net, batch_X, batch_y)
            loss.backward()
            optimizer.step()
        loss_lst.append(loss)
        acc_lst.append(acc)
        print(f"Epoch: {epoch}. Loss: {loss}. Accuracy: {acc}")
    save_model_to_validate(net, loss_lst[-1], acc_lst[-1])
    return loss_lst, acc_lst

def validate_compute_accuracy(net, X_test1, Y_test1):
    '''
        Compute the accuracy scores given X and y.
    '''
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

def save_model_to_validate(model, end_loss, end_acc):
    '''
        Save the model to validate the model for later use.
    '''
    torch.save(model, PATH)
    print("Model Saved at End Loss: {} and End Acc: {}!".format(end_loss, end_acc))


def caller():
    '''
        Pass the data and train the neural network.
        Train for both cases, test for both cases.
    '''
    X_train1, X_test1, Y_train1, Y_test1, \
        X_train2, X_test2, \
        Y_train2, Y_test2 = preprocess_data.read_prep_data()
    print("Read data!")
    print("X_train1, X_train2: ", X_train1.shape, X_train2.shape)
    print("Y_train1, Y_train2: ", Y_train1.shape, Y_train2.shape)
    print("X_test1, X_test2: ", X_test1.shape, X_test2.shape)
    print("Y_test1, Y_test2: ", Y_test1.shape, Y_test2.shape)
    
    X_train1 = torch.from_numpy(X_train1).float()
    X_train2 = torch.from_numpy(X_train2).float()
    X_test1 = torch.from_numpy(X_test1).float()
    X_test2 = torch.from_numpy(X_test2).float()
    
    Y_train1 = torch.from_numpy(Y_train1).float()
    Y_test1 = torch.from_numpy(Y_test1).float()
    Y_train2 = torch.from_numpy(Y_train2).float()
    Y_test2 = torch.from_numpy(Y_test2).float()

    net = Net()
    print("Training for first case: Given X, predict if death or no death!")
    loss_lst_1, acc_lst_1 = train_net(net, X_train1, Y_train1)
    print("Test Accuracy Case # 01: ", validate_compute_accuracy(net, X_test1, Y_test1))
    print("Training for second case: Given X, predict if recovered or no recovered!")
    loss_lst_2, acc_lst_2 = train_net(net, X_train2, Y_train2)
    print("Test Accuracy Case # 02: ", validate_compute_accuracy(net, X_test2, Y_test2))

    print("Graphing the accuracy and loss plots.")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axes[0].plot(loss_lst_1)
    axes[0].plot(acc_lst_1)
    axes[1].plot(loss_lst_2)
    axes[1].plot(acc_lst_2)
    axes[0].title.set_text("Acc and Loss for Case # 01")
    axes[1].title.set_text("Acc and Loss for Case # 02")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    caller()