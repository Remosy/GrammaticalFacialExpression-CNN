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
data = np.load(Normalisation_location + 'topics_datapoints.npy')
np.random.shuffle(data)
# chars = list(set(data))
# data_size, vocab_size = len(data), len(chars)
# print('data has %d characters, %d unique.' % (data_size, vocab_size))
# char_to_ix = {ch: i for i, ch in enumerate(chars)}
# ix_to_char = {i: ch for i, ch in enumerate(chars)}

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
INPUT_SIZE = 118  # 11 vectors
TOTAL_SIZE = len(train_input)
OUTPUT_SIZE = 2
HIDDEN_SIZE =105  # size of hidden layer of neurons
NUM_LAYER = 1  #
NUM_STEP = 50  # number of steps to unroll the RNN for
learning_rate = 0.0099
num_epochs = 20


##-------------------------------------------------
# LSTM Models
##-------------------------------------------------
class LSTM_GFE(nn.Module):
    def __init__(self):
        super(LSTM_GFE, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYER,
            #batch_first=True,
        )
        self.FC = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, input,  hprev, cprev):
        input = Variable(torch.cat(input.data).view(len(input), 1, -1))
        #input = input.contiguous().view(input.data.shape[0], 1, input.data.shape[1])
        #print("Continue: "+str(input.shape))

        output, hc = self.lstm(input, (hprev, cprev))
        output = self.FC(output[:, -1, :])
        #print(output.shape)
        return output.view(output.shape[0],output.shape[1]), hc


##-------------------------------------------------
# TRAINING
##-------------------------------------------------
def TrainModel(Model):
    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)

    all_train_losses = []
    all_test_losses = []

    print(X_train.shape)
    ct = 0
    # exit(0)

    for epoch in range(num_epochs):
        train_ct = 0
        hprev = Variable(torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE))  # reset RNN memory
        cprev = Variable(torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE))

        Y_predict, hc = Model(X_train,hprev, cprev)
        hprev = hc[0]
        cprev = hc[1]
        hprev.detach_()
        cprev.detach_()
        loss = loss_func(Y_predict, Y_train)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


        all_train_losses.append(loss.data[0])
        Y_pred_test, hc = Model(X_test,hprev, cprev)
        loss_test = loss_func(Y_pred_test, Y_test)
        all_test_losses.append(loss_test.data[0])
        if epoch %3 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test loss: %.4f' % loss_test.data[0])


    import matplotlib.pyplot as plt

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.plot(all_train_losses, label='Training')
    plt.plot(all_test_losses, color="orange", label='Testing')
    plt.legend()
    plt.show()


##-------------------------------------------------
# MODEL Evaluation
##-------------------------------------------------
def EvaModel(Model):
    hprev = Variable(torch.zeros(1, 1, HIDDEN_SIZE))  # reset RNN memory
    cprev = Variable(torch.zeros(1, 1, HIDDEN_SIZE))

    confusion = torch.zeros(OUTPUT_SIZE, OUTPUT_SIZE)
    Y_predict,hc = Model(X_train,hprev, cprev)
    _, predicted = torch.max(Y_predict, 1)

    for i in range(train_data.shape[0]):
        actual_class = Y_train.data[i]
        predicted_class = predicted.data[i]

        confusion[actual_class][predicted_class] += 1
    print('')
    print('Confusion matrix for training 1:')
    print(confusion)

    Y_pred_test, hc = Model(X_test,hprev, cprev)
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

    tp = confusion_test[1, 1]
    fp = confusion_test[0, 1]
    fn = confusion_test[1, 0]
    F1 = 2*tp/(2*tp+fp+fn)
    print('F1-Score: %.4f %%' % F1)

##-------------------------------------------------
# Start
##-------------------------------------------------

net = LSTM_GFE()
print(net)
TrainModel(net)
EvaModel(net)





