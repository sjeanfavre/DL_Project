import torch
import matplotlib.pyplot as plt

import utility_functions
from utility_functions import train_and_test


mini_batch_size1 = 50                               # ConvNet
mini_batch_size2 = 50                                # MLP
nb_epochs1 = 50                                      # ConvNet
nb_epochs2 = 25                                      # MLP
nb_hidden = 10                                       # Hidden units last layer ConvNet
nb_hidden1 = 256                                     # Hidden units first layer MLP
nb_hidden4 = 100                                     # Hidden units last layer MLP
lr = 0.001                                           # Learning rate
nb_rounds = 20
nsamples = 1000
nvariations = 4
loss1 = torch.zeros((nvariations, nb_epochs1))    # ConvNet
loss2 = torch.zeros((nvariations, nb_epochs2))    # MLP
model_variations = ['Original', 'WS', 'AL', 'WS-AL']
plots = True

# Compare performance of the two architectures
# ConvNet
for model_n in range(nvariations):
    
    loss1[model_n] = train_and_test(model_n, mini_batch_size1, nb_epochs1, nb_hidden, None,
                                    None, lr, nb_rounds, nsamples)
    
    if plots:  # Generate and save plots

        plt.plot(range(nb_epochs1), loss1[model_n], label=model_variations[model_n])
        plt.xlabel('Epoch')
        plt.ylabel('Average loss')
        plt.title('ConvNet - ' + model_variations[model_n])
        plt.legend()
        plt.savefig('plots/convnet_loss_' + model_variations[model_n] + '.png')

        if model_n == nvariations-1:
            for i in range(nvariations):
                plt.plot(range(nb_epochs1), loss1[i], label=model_variations[i])
            plt.xlabel('Epoch')
            plt.ylabel('Average loss')
            plt.title('ConvNet')
            plt.legend()
            plt.savefig('plots/convnet_loss_all.png')
                
    
    
# MLP
for model_n in range(nvariations):
    
    loss2[model_n] = train_and_test(model_n+4, mini_batch_size2, nb_epochs2, None, nb_hidden1,
                                    nb_hidden4, lr, nb_rounds, nsamples)
    
    if plots:  # Generate and save plots

        plt.plot(range(nb_epochs2), loss2[model_n], label=model_variations[model_n])
        plt.xlabel('Epoch')
        plt.ylabel('Average loss')
        plt.title('MLP - ' + model_variations[model_n])
        plt.legend()
        plt.savefig('plots/mlp_loss_' + model_variations[model_n] + '.png')

        if model_n == nvariations-1:
            
            for i in range(nvariations):
                plt.plot(range(nb_epochs2), loss2[i], label=model_variations[i])
                
            plt.xlabel('Epoch')
            plt.ylabel('Average loss')
            plt.title('MLP')
            plt.legend()
            plt.savefig('plots/mlp_loss_all.png')    
     