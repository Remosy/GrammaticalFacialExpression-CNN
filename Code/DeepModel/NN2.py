
import numpy as np
from numpy import (array, dot, arccos, clip)
from numpy.linalg import norm
import torch
import torch.nn.functional as F
import operator
from torch.autograd import Variable
Normalisation_location = "../../RMS/ab_numpy/data_norm/"
data = np.load(Normalisation_location+'yn_question_datapoints.npy')

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

class Pruned_NN(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, update_weight):
        super(Pruned_NN, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        self.out.weight.data = torch.Tensor(update_weight).float()

    def forward(self, x):
        h_input = self.hidden(x)
        h_output = F.sigmoid(h_input)
        y_pred = self.out(h_output)
        return y_pred

def TrainModel(Model):
    loss_func = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(Model.parameters(), lr=learning_rate)

    all_train_losses = []
    all_test_losses = []

    for epoch in range(num_epochs):
        Y_predict = Model(X_train)
        loss = loss_func(Y_predict, Y_train)
        all_train_losses.append(loss.data[0])
        Model.zero_grad()
        loss.backward()
        optimiser.step()

        Y_pred_test = Model(X_test)
        loss_test = loss_func(Y_pred_test, Y_test)
        all_test_losses.append(loss_test.data[0])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(all_train_losses)
    plt.plot(all_test_losses, color="green")
    plt.show()
##-------------------------------------------------
# Train the neural network to get the 1st NN
##-------------------------------------------------
net = NN_GFE(input_neurons, hidden_neurons, output_neurons)
TrainModel(net)
##-------------------------------------------------
# Get redundant neurons
##-------------------------------------------------

max_range = int(hidden_neurons)
list_size = int(np.math.factorial(max_range) / (np.math.factorial(2) * np.math.factorial(int(max_range - 2))))
Hidden_Input = np.array(net.hidden.weight.data,np.float32)
Hidden_Output = np.array(net.out.weight.data,np.float32)

num_list = set(range(0,max_range))

def getCombination(iter, r):
    space = tuple(iter)
    n = len(space)
    if r > n:
        return
    indice = list(range(r))
    yield tuple(space[i] for i in indice)
    while True:
        for i in reversed(list(range(r))):
            if indice[i] != i + n - r:
                break
        else:
            return
        indice[i] += 1
        for j in list(range(i+1, r)):
            indice[j] = indice[j - 1] + 1
        yield tuple(space[i] for i in indice)

comb_list = getCombination(num_list,2) #get (1,2)(2,3)(3,4)....combinations

def getAngle(h1,h2):

    v1 = Hidden_Output[:,h1]
    v2 = Hidden_Output[:,h2]
    v12 = dot(v1,v2)/norm(v1)/norm(v2)
    return np.degrees(arccos(clip(v12,-1,1)))


#print(list_size)
#print(num_list)
bucket = dict()
for xx in comb_list:
    bucket[xx] = getAngle(xx[0],xx[1])


sorted_x = sorted(bucket.items(), key=operator.itemgetter(1))

trash_bin = set([])
for yy in sorted_x:
    #print(yy)
    if yy[1] < 5 or yy[1]>175:
        trash_bin.add(yy[0][0])

hidden_neurons = hidden_neurons - len(trash_bin)


print(trash_bin)
print(num_list)
indic = num_list - trash_bin
print(indic)
new_weight = np.zeros(shape=[2,hidden_neurons])
ct = 0
for zz in indic:
    new_weight[:,ct] = Hidden_Output[:,zz]
    ct += 1
print("Combination ===>"+str(list_size))
print("New Weight===>")
print(new_weight)

##-------------------------------------------------
# Update weight (hidden->output) to retrain
##-------------------------------------------------
net2 = Pruned_NN(input_neurons, hidden_neurons, output_neurons, new_weight)
TrainModel(net2)

##-------------------------------------------------
# Evaluations Model 1
##-------------------------------------------------

confusion = torch.zeros(output_neurons, output_neurons)
Y_predict = net(X_train)
_, predicted = torch.max(Y_predict, 1)

for i in range(train_data.shape[0]):
    actual_class = Y_train.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for training 1:')
print(confusion)

Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('Testing 1 Accuracy: %.4f %%' % (correct_test / total_test))

confusion_test = torch.zeros(output_neurons, output_neurons)
for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing 1:')
print(confusion_test)

##-------------------------------------------------
# Evaluations Model 2
##-------------------------------------------------
confusion = torch.zeros(output_neurons, output_neurons)
Y_predict = net2(X_train)
_, predicted = torch.max(Y_predict, 1)

for i in range(train_data.shape[0]):
    actual_class = Y_train.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1
print('')
print('Confusion matrix for training 2:')
print(confusion)


Y_pred_test = net2(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing 2 Accuracy: %.4f %%' % (correct_test / total_test))


confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing 2:')
print(confusion_test)

