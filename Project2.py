import math, torch


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

class Parameters(object):
    # Class defining the parameters used in the neural network and their gradient
    def __init__(self, *size):
        self.value = torch.empty(size).normal_(0,1)
        self.gradient = torch.empty(size).fill_(0)
    def get_tuple(self):
        return self.value, self.gradient

class Linear(Module):
    # Fully connected layer module   
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = Parameters(in_features, out_features)
        self.b = Parameters(out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.param = [self.w, self.b]
        
    def forward(self, inputt):
        self.inputt = inputt    #Saving for backprop
        return self.inputt @ self.w.value + self.b.value#torch.mm(self.w, self.inputt) + self.b
    
    def backward(self, dl_ds):
        self.w.gradient = torch.mm(dl_ds,torch.transpose(self.x,0,1))
        self.b.gradient = dl_ds
        return torch.mm(torch.transpose(self.w,0,1), dl_ds)
    
    def param(self):
        return [(self.w,self.w.gradient), (self.b,self.w.gradient)]


class LossMSE(object):
    # LossMSE module
    def loss(self, v, t):
        # Computes the MSE loss of v and target t
        return torch.mean((v-t).pow(2))
    
    def gradient(self, v, t):
        # Computes the MSE loss gradient w.r.t. v
        return 2*(v-t)/v.numel()   


#Activation functions

class ReLU(Module):
    def __init__(self, *size):
        super().__init__()
        
        self.inputt = torch.empty(size)
        self.output = torch.empty(size)
        self.param = []
        
    def forward(self, s):
        self.s = s
        s[s>0] = 0
        return s
    
    def backward(self, dl_dx):
        dl_ds = dl_dx*(self.s>0)
        return dl_ds
    
class Tanh(Module):
    def __init__(self, *size):
        super().__init__()
        self.inputt = torch.empty(size)
        self.output = torch.empty(size)
        self.param = []
        
    def forward(self, s):
        self.s = s
        self.output = [math.tanh(x) for x in self.s]
        return torch.Tensor(self.output)

    def backward(self, dl_dx):
        d_tanh = 1-((1-math.exp(-2*self.s))/(1+math.exp(-2*self.s)))^2
        dl_ds = dl_dx*d-tanh
        return dl_ds

    
#Sequential
class Sequential(Module):
    def __init__(self, *structure):
        super().__init__()
        self.param = []
        self.structure = structure
        
    def forward(self, x):
        for layer in self.structure:
            x = layer.forward(x)
        
    def backward(self, dl_dx):
        for layer in reversed(self.structure):
            dl_dx = layer.backward(dl_dx)
    
    def param(self):
        parameter = []
        for layer in self.structure:
            for parameter in layer.param():
                parameters.append(parameter)

class SGD():
    # ...
    def __init__(self, param, eta, gamma = 0.5):
        if eta < 0.:
            raise ValueError(f"Method not implemented for $\eta$ < 0")
        self.param = param
        self.eta = eta
        self.gamma = gamma
        #self.wt = []
        #for i in self.param:
            #self.wt.append(i.gradient.clone())

    def step(self):
        for i in range(len(self.param)):
            #Rumelhart et al . 1986
            ut[i] = self.gamma * self.ut[i] + self.eta * self.param[i].gradient
            self.param[i].value = self.wt[i] - ut[i]
            self.param[i].gradient[:].fill_(0)

# Generation of data            

def generate_data():
    # Generates uniformly distributed data points in [0,1]^2, each with a label 0 if outside the disk of center (0.5, 0.5) and radius 1/sqrt(2*pi), and 1 inside
    
    train_input = torch.rand(1000, 2)
    test_input = torch.rand(1000, 2)
    train_target = ((train_input[:,0]-0.5)**2 + (train_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    test_target = ((test_input[:,0]-0.5)**2 + (test_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    
    return train_input, train_target.int(), test_input, test_target.int()