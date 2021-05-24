import math
import torch
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)


class Module(object):
   
    def __init__(self):
        self.param = [] #Initializing the parameters
        
    def forward(self, *inputt):
        '''
        Input
        -----
        *inputt : a tensor or a tuple of tensors.
            
        Output
        ------
        a tensor or a tuple of tensors.
        '''
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        '''
        Input
        -----
        *gradwrtoutput : a tensor or a tuple of tensors 
        containing the gradient wrt to the output
            
        Output
        ------
        a tensor or a tuple of tensors containing the gradient
        of the loss w.r.t. the module’s input.
        '''
        raise NotImplementedError
        
    def param(self):
        '''    
        Output
        ------
        a list of pairs, each composed of a parameter tensor, 
        and a gradient tensor of same size. This list should 
        be empty for parameterless modules (e.g. ReLU).
        '''
        return []


class Linear(Module):
    
    def __init__(self, in_features, out_features, gamma = 1e-3):
        self.gamma = gamma
        self.w = torch.empty(out_features, in_features).normal_()
        self.b = torch.empty(out_features, 1).normal_()

    def forward(self, x):
        #Save input for backprop
        self.x = x    
        return torch.mm(self.w,x) + self.b
    
    def backward(self, dl_ds):
        #Derivative of the loss w.r.t. the parameters
        
        self.dl_dw = torch.mm(dl_ds,self.x.T)
        self.dl_db = torch.sum(dl_ds,1).unsqueeze(1)   #Sum the gradients of the batch
        self.dl_dx = torch.mm(self.w.T,dl_ds)
        #Parameters update
        self.w -= self.gamma*self.dl_dw
        self.b -= self.gamma*self.dl_db
        
        return self.dl_dx
    
    def param(self):
        return [(self.w,self.dl_dw), (self.b,self.dl_db)]
    
#Activation functions

class ReLU(Module):

    def forward(self, s):
        self.s = s
        s[s>0] = 0
        return s

    def backward(self, dl_dx):
        dl_ds = dl_dx*(self.s>0)
        return dl_ds
    
    
class Tanh(Module):
    
    def forward(self, s):
        self.s = s
        return (1-torch.exp(-2*s))/(1+torch.exp(-2*s))

    def backward(self, dl_dx):
        d_tanh = 1-((1-torch.exp(-2*self.s))/(1+torch.exp(-2*self.s)))**2
        dl_ds = dl_dx*d_tanh
        return dl_ds


#Sequential
class Sequential(Module):
    def __init__(self, structure):
        self.structure = structure
        
    def forward(self, x):
        for layer in self.structure:
            x = layer.forward(x)
        return x
        
    def backward(self, dl_dx):
        for layer in reversed(self.structure):
            dl_dx = layer.backward(dl_dx)
        return dl_dx
    
    def param(self):
        parameters = []
        for layer in self.structure:
            for parameter in layer.param():
                print(len(layer.param()))
                parameters.append(parameter)
        return parameters
            
#Loss function
class LossMSE(object):
    def loss(self, v, t):
        # Computes the MSE loss of v and target t
        return torch.mean((v-t).pow(2))
    
    def gradient(self, v, t):
        # Computes the MSE loss gradient w.r.t. v
        return 2*(v-t)/v.numel()   


def generate_data():
    # Generates uniformly distributed data points in [0,1]^2, each with a label 0 if outside the disk of center (0.5, 0.5) and radius 1/sqrt(2*pi), and 1 inside
    
    train_input = torch.rand(1000, 2)
    test_input = torch.rand(1000, 2)
    train_target = ((train_input[:,0]-0.5)**2 + (train_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    test_target = ((test_input[:,0]-0.5)**2 + (test_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    
    return train_input, train_target.int(), test_input, test_target.int()


#Parameters
nb_epochs = 200
epochs = range(nb_epochs)
eta = 0.001
gamma = 0
batch_size = 25

#Data generation
train_input, train_target, test_input, test_target = generate_data()

train_errors = []
train_loss = []
test_errors = []
test_loss = []

#Network
model = Sequential([Linear(in_features = 2, out_features = 25),
                    Tanh(),
                    Linear(in_features = 25, out_features = 25),
                    Tanh(),
                    Linear(in_features = 25, out_features = 25),
                    Tanh(),
                    Linear(in_features = 25, out_features = 2),
                    ReLU()])

criterion = LossMSE()
loss = []

for e in range(nb_epochs):  

    for b in range(0, train_input.size(0), batch_size):
        #Forward pass
        output = model.forward(train_input[b:b+batch_size].T)
        #Store loss
        loss.append(criterion.loss(output, train_target[b:batch_size]))
        #Backpropagation
        model.backward(criterion.gradient(output, train_target[b:batch_size]))
        #Parameters update
        for p in model.param():
            p -= eta * p.grad
    
    
    
    
"""    
#from lecture 5.2 SGD
for e in range(nb_epochs):
    model.zero_grad()
    for b in range(0, train_input.size(0), batch_size):
        output = model(train_input[b:b+batch_size])
        loss = criterion(output, train_target[b:b+batch_size])
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
"""