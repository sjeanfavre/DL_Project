from structure import *
from utils import *
import matplotlib.pyplot as plt


#Parameters setting
nb_epochs = 2000
epochs = range(nb_epochs)
eta = 0.1
gamma = 1e-3
batch_size = 50

#Data generation
train_input, train_target, test_input, test_target = generate_data()


#Structure of the network
model = Sequential([Linear(in_features = 2, out_features = 25),
                    ReLU(),
                    Linear(in_features = 25, out_features = 25),
                    Tanh(),
                    Linear(in_features = 25, out_features = 25),
                    Tanh(),
                    Linear(in_features = 25, out_features = 1),
                    Tanh()])

#Loss
criterion = LossMSE()

#Initialization
train_error = torch.empty(train_target.size())
test_error = torch.empty(train_target.size())
output_test_all = torch.empty(train_target.size())
nb_train_errors = torch.zeros(nb_epochs)
nb_test_errors = torch.zeros(nb_epochs)

for e in range(nb_epochs):  

    for b in range(0, train_input.size(0), batch_size):
        #Forward pass
        output_test = model.forward(test_input[b:b+batch_size].T)
        output_train = model.forward(train_input[b:b+batch_size].T)
        
        output_test_all[b:b+batch_size] = output_test
        #Error
        nb_train_errors[e] += compute_nb_error(output_train, train_target[b:b+batch_size])
        nb_test_errors[e] += compute_nb_error(output_test, test_target[b:b+batch_size])
        #Backpropagation
        model.backward(criterion.grad(output_train, train_target[b:b+batch_size].T))
        
    print("Epoch {}".format(e+1),end='\r')
    

#If prediction > 0.5, it it set to 1, otherwise to 0
output_test_all = (output_test_all >= 0.5).int()


#Plot


plot(test_input, output_test_all, nb_epochs, nb_train_errors, nb_test_errors)


