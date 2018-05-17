"""
Minimal character-level Pytorch LSTM model. Written by Jo Plested based on vanilla RNN by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

##-------------------------------------------------
# DATA Preparation
##-------------------------------------------------
Normalisation_location = "../data_norm/"
data = np.load(Normalisation_location+'yn_question_datapoints.npy')
np.random.shuffle(data)
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)
#print('data has %d characters, %d unique.' % (data_size, vocab_size))
#char_to_ix = {ch: i for i, ch in enumerate(chars)}
#ix_to_char = {i: ch for i, ch in enumerate(chars)}

msk = np.random.rand(len(data)) < 0.8

train_data = data[msk]
test_data = data[~msk]
n_attributes = train_data.shape[1] - 1

train_input = train_data[:, :n_attributes]
train_target = train_data[:, n_attributes]

test_input = test_data[:, :n_attributes]
test_target = test_data[:, n_attributes]

X_train = Variable(torch.Tensor(train_input).float())
Y_train = Variable(torch.Tensor(train_target).long())

X_test = Variable(torch.Tensor(test_input).float())
Y_test = Variable(torch.Tensor(test_target).long())

##-------------------------------------------------
# LSTM Hyperparameters
##-------------------------------------------------
INPUT_SIZE = 118 #11 vectors
OUTPUT_SIZE = 2
HIDDEN_SIZE = 100  # size of hidden layer of neurons
NUM_LAYER = 1 #
NUM_STEP = 50  # number of steps to unroll the RNN for
learning_rate = 1e-1
num_epochs = 10
##-------------------------------------------------
# LSTM Models
##-------------------------------------------------
class LSTM_GFE(nn.Module):
    def __init__(self):
        super(LSTM_GFE, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size = HIDDEN_SIZE,
            num_layers=NUM_LAYER,
            batch_first=True,
        )
        self.FC = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, input,hprev):
        output, hc = self.lstm(input, hprev)
        output = self.FC(output)
        return output.view(INPUT_SIZE), hc
##-------------------------------------------------
# TRAINING
##-------------------------------------------------
def TrainModel(Model, h, c, num_seq):
    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)

    all_train_losses = []
    all_test_losses = []
    hc = None;

    #ToDo: input Size????
    
    print(X_train.shape)
    exit(0)
    for epoch in range(num_epochs):
        Y_predict, hprev = Model(X_train,hc)
        hc = Variable(hprev.data) #Update hc
        loss = loss_func(Y_predict, Y_train)
        all_train_losses.append(loss.data[0])
        Model.zero_grad()
        loss.backward()
        optimiser.step()

        Y_pred_test = Model(X_test,hc)
        loss_test = loss_func(Y_pred_test, Y_test)
        all_test_losses.append(loss_test.data[0])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(all_train_losses)
    plt.plot(all_test_losses, color="green")
    plt.show()
##-------------------------------------------------
# MODEL Evaluation
##-------------------------------------------------
def EvaModel(Model):
    confusion = torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE)
    Y_predict = Model(X_train)
    _, predicted = torch.max(Y_predict, 1)

    for i in range(train_data.shape[0]):
        actual_class = Y_train.data[i]
        predicted_class = predicted.data[i]

        confusion[actual_class][predicted_class] += 1
    print('')
    print('Confusion matrix for training 1:')
    print(confusion)

    Y_pred_test = Model(X_test)
    _, predicted_test = torch.max(Y_pred_test, 1)
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
    print('Testing Accuracy: %.4f %%' % (correct_test / total_test))

    confusion_test = torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE)
    for i in range(test_data.shape[0]):
        actual_class = Y_test.data[i]
        predicted_class = predicted_test.data[i]

        confusion_test[actual_class][predicted_class] += 1

    print('')
    print('Confusion matrix for testing 1:')
    print(confusion_test)



net = LSTM_GFE()
print(net)
TrainModel(net,1,1,1)
EvaModel(net)





