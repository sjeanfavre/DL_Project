import torch.empty, math

from torch import empty

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
    def __init__(self, in_size, out_size):
        self.w = torch.rand(in_size, out_size)
        self.b = torch.rand(out_size, 1)
        