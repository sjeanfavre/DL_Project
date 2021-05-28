import math
import torch
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

#Seed the random number generator for reproducability
torch.manual_seed(1)

#Parent module
class Module(object):
   
    def __init__(self):
        self.param = []
        
    def forward(self, *inputt):
        raise NotImplementedError
        
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
        
    def param(self):
        return []

#Fully connected layer
class Linear(Module):
    
    def __init__(self, in_features, out_features, gamma = 1e-3):
        self.gamma = gamma
        self.w = torch.empty(out_features, in_features).normal_()
        self.b = torch.empty(out_features, 1).normal_()
        self.dl_dw = None
        self.dl_db = None
        
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
        return [[self.w,self.dl_dw], [self.b,self.dl_db]]
    
#Activation functions
class ReLU(Module):

    def forward(self, s):
        self.s = s
        s[s<0] = 0
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
        return [layer.param() for _, layer in enumerate(self.structure)]
    
#Loss function
class LossMSE(object):
    def loss(self, v, t):
        # Computes the MSE loss of v and target t
        return torch.mean((v-t).pow(2))
    
    def grad(self, v, t):
        # Computes the MSE loss gradient w.r.t. v
        return 2*(v-t)/v.numel()   
    
class LossMAE(object):
    def loss(self, v, t):
        # Computes the MSE loss of v and target t
        return torch.mean(abs(v-t))
    
    def grad(self, v, t):
        # Computes the MSE loss gradient w.r.t. v
        return 2*(v-t)/v.numel()   
