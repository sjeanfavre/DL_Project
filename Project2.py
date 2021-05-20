import math
from torch import empty, set_grad_enabled

set_grad_enabled(False)

class Module(object):
    
    def __init__(self):
        self.parameters = [] #Initializing the parameters
        
    def forward(self, *input):
        '''
        Input
        -----
        *input : a tensor or a tuple of tensors.
            
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
    # Fully connected layer module
    
    def __init__(self, in_size, out_size):
        self.w = torch.rand(in_size, out_size)
        self.b = torch.rand(out_size, 1)
        


class LossMSE(Module):
    # LossMSE module
    
    def forward(self, v, t):
        # Computes the MSE loss of v and target t
        
        self.v = v
        self.t = t
        return torch.mean((v-t).pow(2))
    
    def backward(self, v, t):
        # Computes the MSE loss gradient w.r.t. v
        
        return 2*(self.v-self.t)/v.numel()    
    
    
        
def generate_data():
    # Generates uniformly distributed data points in [0,1]^2, each with a label 0 if outside the disk of center (0.5, 0.5) and radius 1/sqrt(2*pi), and 1 inside
    
    train_input = torch.rand(1000, 2)
    test_input = torch.rand(1000, 2)
    train_target = ((train_input[:,0]-0.5)**2 + (train_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    test_target = ((test_input[:,0]-0.5)**2 + (test_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    
    return train_input, train_target, test_input, test_target


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
        return (1-math.exp(-2*s))/(1+math.exp(-2*s))

    def backward(self, dl_dx):
        d_tanh = 1-((1-math.exp(-2*self.s))/(1+math.exp(-2*self.s)))^2
        dl_ds = dl_dx*d-tanh
        return dl_ds