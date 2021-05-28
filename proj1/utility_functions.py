import torch
from torch import nn
from torch import optim
from time import perf_counter 

import dlc_practical_prologue as prologue
from architectures import ConvNet, ConvNet_WS, ConvNet_AL, ConvNet_WS_AL, \
                          MLP, MLP_WS, MLP_AL, MLP_WS_AL



def size_out_conv2d(Nin, Cin, Hin, Win, Cout, padding, ker, stride):
    #considering dilation=1
    Hout = ((Hin + 2*padding - ker)/stride + 1)//1
    Wout = ((Win + 2*padding - ker)/stride + 1)//1
    return Nin, Cout, Hout, Wout

###########################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###########################################################################################

def train_model(model, train_input, train_target, mini_batch_size, nb_epochs, lr, al=False, train_classes=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    loss_epoch = torch.zeros((nb_epochs))

    for e in range(nb_epochs):
        acc_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            if al: # Use auxiliary loss
                output, out_im1, out_im2 = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                loss_im1 = criterion(out_im1, train_classes[:, 0].narrow(0, b, mini_batch_size))
                loss_im2 = criterion(out_im2, train_classes[:, 1].narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item() + loss_im1.item() + loss_im2.item()
                loss = loss + loss_im1 + loss_im2

            else:
                output = model(train_input.narrow(0, b, mini_batch_size))
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
                acc_loss = acc_loss + loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()

        loss_epoch[e] = acc_loss
    
    return loss_epoch

###########################################################################################

def compute_nb_errors(model, inputt, target, mini_batch_size, al=False):
    nb_errors = 0
    for b in range(0, inputt.size(0), mini_batch_size):
        if al:
            output, _, _ = model(inputt.narrow(0, b, mini_batch_size))
        else:
            output = model(inputt.narrow(0, b, mini_batch_size))
        _, pred = output.max(1)
        for k in range(mini_batch_size):
            if target[b + k] != pred[k]:
                nb_errors = nb_errors + 1
    
    return nb_errors

###########################################################################################

def train_and_test(model_n, mini_batch_size, nb_epochs, nb_hidden, nb_h1, nb_h4, lr, nb_rounds, nsamples):

    nb_test_errors = torch.zeros((nb_rounds))
    nb_train_errors = torch.zeros((nb_rounds))
    train_time = torch.zeros((nb_rounds))
    loss = torch.zeros((nb_rounds, nb_epochs))
    a_l = False
    
    # Load data
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nsamples)

    for i in range(nb_rounds): 
        # Randomize data at each round
        perm = torch.randperm(nsamples)
        train_input = train_input[perm]
        train_target = train_target[perm]
        train_classes = train_classes[perm]
        perm = torch.randperm(nsamples)
        test_input = test_input[perm]
        test_target = test_target[perm]
        test_classes = test_classes[perm]
        
        # Load model at each round with randomized weights
        if model_n==0:
            model = ConvNet(nb_hidden)
        elif model_n==1:
            model = ConvNet_WS(nb_hidden)
        elif model_n==2:
            model = ConvNet_AL(nb_hidden)
            a_l = True
        elif model_n==3:
            model = ConvNet_WS_AL(nb_hidden)
            a_l = True
        elif model_n==4:
            model = MLP(nb_h1, nb_h4)
        elif model_n==5:
            model = MLP_WS(nb_h1, nb_h4)
        elif model_n==6:
            model = MLP_AL(nb_h1, nb_h4)
            a_l = True
        elif model_n==7:
            model = MLP_WS_AL(nb_h1, nb_h4)
            a_l = True
        t1_start = perf_counter()
        loss[i] = train_model(model, train_input, train_target, mini_batch_size, nb_epochs, lr, a_l, train_classes)
        t1_stop = perf_counter()
        train_time[i] = t1_stop-t1_start
        nb_test_errors[i] = compute_nb_errors(model, test_input, test_target, mini_batch_size, a_l)
        nb_train_errors[i] = compute_nb_errors(model, train_input, train_target, mini_batch_size, a_l)

    print(model)    
    print('Number of parameters:', count_parameters(model))
    print('Training error: {:0.2f}% ± {:0.2f}%'.format(100 * nb_train_errors.mean() / nsamples,
                                                       100 * nb_train_errors.std() / nsamples))
    print('Test error: {:0.2f}% ± {:0.2f}%'.format(100 * nb_test_errors.mean() / nsamples,
                                                   100 * nb_test_errors.std() / nsamples))
    print('Training time: {:0.2f} s/epoch'.format(train_time.mean() / nb_epochs))
    print('\n')

    
    return loss.mean(0)

###########################################################################################

# Perform several rounds of K-fold Cross validation
def cross_validation_nrounds(nrounds, k=10):
    
    nsamples = 1000
    nmodels = 12
    nsamples_red = int(nsamples / k)
    batch_and_epochs = [[25, 25], [50, 25], [100, 25],
                        [25, 50], [50, 50], [100, 50]]
    nb_hidden = 10
    nb_hidden1 = 256
    nb_hidden4 = 100
    lr = 0.001

    train_err = torch.zeros((nmodels, nrounds, k))
    test_err = torch.zeros((nmodels, nrounds, k))
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(nsamples)

    idx = torch.arange(0,k)

    for l in range(nrounds):
        # Randomize data at each round
        perm = torch.randperm(nsamples)
        train_input = train_input.view(nsamples, 2, 14, 14)[perm]
        train_target = train_target.view(nsamples)[perm]

        train_input = train_input.view(k, nsamples_red, 2, 14, 14)
        train_target = train_target.view(k, nsamples_red)
        
        for i in range(k):
            # Form C-V train and test sets
            test_in_cv = train_input[i]
            test_target_cv = train_target[i]
            train_in_cv = train_input[idx!=i].view(-1, 2, 14, 14)
            train_target_cv = train_target[idx!=i].view(-1)
            
            # Train each model
            for j in range(6):
                archi1 = ConvNet(nb_hidden)
                _ = train_model(archi1, train_in_cv, train_target_cv, batch_and_epochs[j][0], batch_and_epochs[j][1], lr)
                train_err[j][l][i] = compute_nb_errors(archi1, train_in_cv,
                                                       train_target_cv, batch_and_epochs[j][0]) * 100 / train_in_cv.size(0)
                test_err[j][l][i] = compute_nb_errors(archi1, test_in_cv,
                                                      test_target_cv, batch_and_epochs[j][0]) * 100 / train_in_cv.size(0)

                archi2 = MLP(nb_hidden1, nb_hidden4)
                _ = train_model(archi2, train_in_cv, train_target_cv, batch_and_epochs[j][0], batch_and_epochs[j][1], lr)
                train_err[j+6][l][i] = compute_nb_errors(archi2, train_in_cv,
                                                         train_target_cv, batch_and_epochs[j][0]) * 100 / train_in_cv.size(0)
                test_err[j+6][l][i] = compute_nb_errors(archi2, test_in_cv, 
                                                        test_target_cv, batch_and_epochs[j][0]) * 100 / train_in_cv.size(0)


    # Print results
    print(str(k)+'-fold Cross Validation on', nrounds, 'rounds')
    for j in range(6):
        print('Model: ConvNet -', 'Nb epochs:', batch_and_epochs[j][1], '- Batch size:', batch_and_epochs[j][0])
        print('Training error: {:0.2f}% ± {:0.2f}%  Test error: {:0.2f}% ± {:0.2f}%'.format(train_err[j].mean(),
                                                                                            train_err[j].std(),
                                                                                            test_err[j].mean(),
                                                                                            test_err[j].std()))
        print('---------------------------------------------------------------------------')

    for j in range(6):
        print('Model: MLP -', 'Nb epochs:', batch_and_epochs[j][1], '- Batch size:', batch_and_epochs[j][0])
        print('Training error: {:0.2f}% ± {:0.2f}%  Test error: {:0.2f}% ± {:0.2f}%'.format(train_err[j+6].mean(),
                                                                                            train_err[j+6].std(),
                                                                                            test_err[j+6].mean(),
                                                                                            test_err[j+6].std()))
        print('---------------------------------------------------------------------------')