import matplotlib.pyplot as plt
import torch
import math


def generate_data():
# Generates uniformly distributed data points in [0,1]^2, each with a label 0 if outside the disk of center (0.5, 0.5) and radius 1/sqrt(2*pi), and 1 inside
    
    train_input = torch.rand(1000, 2)
    test_input = torch.rand(1000, 2)
    train_target = ((train_input[:,0]-0.5)**2 + (train_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    test_target = ((test_input[:,0]-0.5)**2 + (test_input[:,1]-0.5)**2) <= 1/(2*math.pi)
    
    return train_input, train_target.int(), test_input, test_target.int()


def compute_nb_error(output, target):
    output = (output >= 0.5).int()
    error = (output != target).int()
    return error.sum()


def plot(points, target, nb_epochs, nb_train_errors, nb_test_errors):
    
    fig, ax = plt.subplots()
    for i in range(points.size()[0]):
        if target[i].item() == 0:
            pointOut = plt.plot(points[i][0], points[i][1], 'ok', label = 'Points outside the disk')
        else:
            pointIn = plt.plot(points[i][0], points[i][1], 'or', label = 'Points inside the disk')
    circle = plt.Circle((0.5,0.5), 1/math.sqrt(2*math.pi), color = 'k', fill = False, label = 'Disk')
    ax.add_patch(circle)
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0]], [labels[0]], loc='best')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig('plotData.png')

    fig, ax = plt.subplots()
    ax.plot(range(nb_epochs), nb_train_errors/10)
    ax.plot(range(nb_epochs), nb_test_errors/10)
    ax.grid()
    ax.set_xlabel("Epochs",)
    ax.set_ylabel("Error in %",)
    ax.legend(['Training error', 'Test error'])
    plt.savefig('Error.png')
    plt.show()
