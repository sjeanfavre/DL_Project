import torch
from torch import nn
from torch.nn import functional as F


###########################################################################################

class ConvNet(nn.Module):
    def __init__(self, nb_hidden=10):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.fc1_1 = nn.Linear(256, 200)
        self.fc2_1 = nn.Linear(200, 10)
        
        self.fc1_2 = nn.Linear(256, 200)
        self.fc2_2 = nn.Linear(200, 10)
        
        self.fc3 = nn.Linear(20, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(F.max_pool2d(self.conv1_1(u), kernel_size=2, stride=2))
        u = F.relu(F.max_pool2d(self.conv2_1(u), kernel_size=2, stride=2))
        u = F.relu(self.fc1_1(u.view(-1, 256)))
        u = F.relu(self.fc2_1(u))
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(F.max_pool2d(self.conv1_2(v), kernel_size=2, stride=2))
        v = F.relu(F.max_pool2d(self.conv2_2(v), kernel_size=2, stride=2))
        v = F.relu(self.fc1_2(v.view(-1, 256)))
        v = F.relu(self.fc2_2(v))
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc3(w.view(-1, 20)))
        w = self.fc4(w)
        
        return w

###########################################################################################

class ConvNet_WS(nn.Module):        #weight sharing
    def __init__(self, nb_hidden=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        self.fc3 = nn.Linear(20, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(F.max_pool2d(self.conv1(u), kernel_size=2, stride=2))
        u = F.relu(F.max_pool2d(self.conv2(u), kernel_size=2, stride=2))
        u = F.relu(self.fc1(u.view(-1, 256)))
        u = F.relu(self.fc2(u))
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(F.max_pool2d(self.conv1(v), kernel_size=2, stride=2))
        v = F.relu(F.max_pool2d(self.conv2(v), kernel_size=2, stride=2))
        v = F.relu(self.fc1(v.view(-1, 256)))
        v = F.relu(self.fc2(v))
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc3(w.view(-1, 20)))
        w = self.fc4(w)
        
        return w

###########################################################################################

class ConvNet_AL(nn.Module):          #auxiliary loss
    def __init__(self, nb_hidden=10):
        super().__init__()
        
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.fc1_1 = nn.Linear(256, 200)
        self.fc2_1 = nn.Linear(200, 10)
        
        self.fc1_2 = nn.Linear(256, 200)
        self.fc2_2 = nn.Linear(200, 10)
        
        self.fc3 = nn.Linear(20, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(F.max_pool2d(self.conv1_1(u), kernel_size=2, stride=2))
        u = F.relu(F.max_pool2d(self.conv2_1(u), kernel_size=2, stride=2))
        u = F.relu(self.fc1_1(u.view(-1, 256)))
        y = self.fc2_1(u)
        u = F.relu(y)
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(F.max_pool2d(self.conv1_2(v), kernel_size=2, stride=2))
        v = F.relu(F.max_pool2d(self.conv2_2(v), kernel_size=2, stride=2))
        v = F.relu(self.fc1_2(v.view(-1, 256)))
        z = self.fc2_2(v)
        v = F.relu(z)
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc3(w.view(-1, 20)))
        w = self.fc4(w)
        
        return w, y, z

###########################################################################################

class ConvNet_WS_AL(nn.Module):          # weight sharing + auxiliary loss
    def __init__(self, nb_hidden=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        self.fc3 = nn.Linear(20, nb_hidden)
        self.fc4 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(F.max_pool2d(self.conv1(u), kernel_size=2, stride=2))
        u = F.relu(F.max_pool2d(self.conv2(u), kernel_size=2, stride=2))
        u = F.relu(self.fc1(u.view(-1, 256)))
        y = self.fc2(u)
        u = F.relu(y)
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(F.max_pool2d(self.conv1(v), kernel_size=2, stride=2))
        v = F.relu(F.max_pool2d(self.conv2(v), kernel_size=2, stride=2))
        v = F.relu(self.fc1(v.view(-1, 256)))
        z = self.fc2(v)
        v = F.relu(z)
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc3(w.view(-1, 20)))
        w = self.fc4(w)
        
        return w, y, z

###########################################################################################

class MLP(nn.Module): 
    def __init__(self, nb_hidden1=256, nb_hidden4=100):
        super().__init__()
        
        self.fc1_1 = nn.Linear(196, nb_hidden1)
        self.fc2_1 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3_1 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc1_2 = nn.Linear(196, nb_hidden1)
        self.fc2_2 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3_2 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc4 = nn.Linear(20, nb_hidden4)
        self.fc5 = nn.Linear(nb_hidden4, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(self.fc1_1(u.view(-1, 196)))
        u = F.relu(self.fc2_1(u))
        u = F.relu(self.fc3_1(u))
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(self.fc1_2(v.view(-1, 196)))
        v = F.relu(self.fc2_2(v))
        v = F.relu(self.fc3_2(v))
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc4(w.view(-1, 20)))
        w = self.fc5(w)
        
        return w

###########################################################################################

class MLP_WS(nn.Module):                                    # weight sharing
    def __init__(self, nb_hidden1=256, nb_hidden4=100):
        super().__init__()
        
        self.fc1 = nn.Linear(196, nb_hidden1)
        self.fc2 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc4 = nn.Linear(20, nb_hidden4)
        self.fc5 = nn.Linear(nb_hidden4, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(self.fc1(u.view(-1, 196)))
        u = F.relu(self.fc2(u))
        u = F.relu(self.fc3(u))
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(self.fc1(v.view(-1, 196)))
        v = F.relu(self.fc2(v))
        v = F.relu(self.fc3(v))
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc4(w.view(-1, 20)))
        w = self.fc5(w)
        
        return w

###########################################################################################

class MLP_AL(nn.Module):                                    # auxiliary loss
    def __init__(self, nb_hidden1=256, nb_hidden4=100):
        super().__init__()
        
        self.fc1_1 = nn.Linear(196, nb_hidden1)
        self.fc2_1 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3_1 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc1_2 = nn.Linear(196, nb_hidden1)
        self.fc2_2 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3_2 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc4 = nn.Linear(20, nb_hidden4)
        self.fc5 = nn.Linear(nb_hidden4, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(self.fc1_1(u.view(-1, 196)))
        u = F.relu(self.fc2_1(u))
        y = self.fc3_1(u)
        u = F.relu(y)
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(self.fc1_2(v.view(-1, 196)))
        v = F.relu(self.fc2_2(v))
        z = self.fc3_2(v)
        v = F.relu(z)
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc4(w.view(-1, 20)))
        w = self.fc5(w)
        
        return w, y, z

###########################################################################################

class MLP_WS_AL(nn.Module):                                    # weight sharing + auxiliary loss
    def __init__(self, nb_hidden1=256, nb_hidden4=100):
        super().__init__()
        
        self.fc1 = nn.Linear(196, nb_hidden1)
        self.fc2 = nn.Linear(nb_hidden1, int(nb_hidden1/2))
        self.fc3 = nn.Linear(int(nb_hidden1/2), 10)
        
        self.fc4 = nn.Linear(20, nb_hidden4)
        self.fc5 = nn.Linear(nb_hidden4, 2)

    def forward(self, x):
        #image1
        u = x[:, 0].view(-1, 1, 14, 14)   
        u = F.relu(self.fc1(u.view(-1, 196)))
        u = F.relu(self.fc2(u))
        y = self.fc3(u)
        u = F.relu(y)
        #image2
        v = x[:, 1].view(-1, 1, 14, 14)
        v = F.relu(self.fc1(v.view(-1, 196)))
        v = F.relu(self.fc2(v))
        z = self.fc3(v)
        v = F.relu(z)
        #concatenate
        w = torch.cat((u,v), 1)
        w = F.relu(self.fc4(w.view(-1, 20)))
        w = self.fc5(w)
        
        return w, y, z

###########################################################################################