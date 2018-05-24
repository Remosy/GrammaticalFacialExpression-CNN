
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
Normalisation_location = "../data_norm/"
data = np.load(Normalisation_location + 'topics_datapoints.npy')

np.random.shuffle(data)
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

input_neurons = n_attributes
hidden_neurons = 15
output_neurons = 2
learning_rate = 0.0005
num_epochs = 1200

class NN_GFE(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(NN_GFE, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = F.sigmoid(h_input)
        y_pred = self.out(h_output)
        return y_pred

net = NN_GFE(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.MSELoss()


optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

all_train_losses = []
all_test_losses = []

for epoch in range(num_epochs):
    Y_predict = net(X_train)
   # print(Y_predict)
    #Y_train = torch.from_numpy(Y_train[:]).float()
    loss = loss_func(Y_predict, Y_train)
    all_train_losses.append(loss.data[0])
    net.zero_grad()
    loss.backward()
    optimiser.step()

    Y_pred_test = net(X_test)
    loss_test = loss_func(Y_pred_test, Y_test)
    all_test_losses.append(loss_test.data[0])


confusion = torch.zeros(output_neurons, output_neurons)

Y_predict = net(X_train)

_, predicted = torch.max(Y_predict, 1)

for i in range(train_data.shape[0]):
    actual_class = Y_train.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)



Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.4f %%' % (correct_test / total_test))


confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_train_losses)
plt.plot(all_test_losses,color ="green")
plt.show()