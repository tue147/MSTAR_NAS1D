import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import VAE_for_3d_graph as vae
import util
from scipy.stats import norm

class Predictor(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_node=13,num_block=1,act=nn.ReLU,dropout=0.2):
        super().__init__()
        self.blocks = nn.Sequential()
        self.blocks.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_block):
            self.blocks.append(act())
            self.blocks.append(nn.Linear(hidden_dim, hidden_dim))
        self.flat = nn.Flatten()
        self.layer2 = nn.Linear(num_node*hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
    def forward(self,z):
        x = self.blocks(z)
        x = self.flat(x)
        x = self.layer2(x)
        x = self.drop(x)
        return self.sig(x)
    
class Predictor_conv(nn.Module):
    def __init__(self,num_node=13,channel=32,act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(4,channel,(num_node,1))
        self.bn = nn.BatchNorm2d(channel)
        self.act = act()
        self.pool = nn.AdaptiveAvgPool2d(1)
        classify = []
        classify.append(nn.Flatten())
        classify.append(nn.Linear(channel,channel//2))
        classify.append(nn.Tanh())
        classify.append(nn.Linear(channel//2,channel//4))
        classify.append(nn.Tanh())
        classify.append(nn.Linear(channel//4,channel//8))
        classify.append(nn.Tanh())
        classify.append(nn.Linear(channel//8,1))
        classify.append(nn.Sigmoid())
        self.classify = nn.Sequential(*classify)
    def forward(self,adj):
        z = self.conv(adj) # shape (bs,channel,num_node,1)
        z = self.bn(z)
        z = self.act(z)
        z = self.pool(z)   # shape (bs,hidden_dim)
        return self.classify(z)
    
class Predictor_linear(nn.Module):
    def __init__(self,channel=2300):
        super().__init__()
        classify = []
        classify.append(nn.Flatten())
        classify.append(nn.Linear(channel,1))

        classify.append(nn.Sigmoid())
        self.classify = nn.Sequential(*classify)
    def forward(self,z):  # shape (bs,channel)
        return self.classify(z)

# Compute the gradient of predictor w.r.t the input x
def grad_ascent(predictor,x,device,lr=1e-2,lr_decay=0.5,epsilon=1e-2,return_grad=False,min_lr=1e-5):
    predictor.to(device)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad = False

    bs, nodes, latent_dim = x.size()  
    x = x.to(device)
    loss = predictor(x)     # shape (bs,1)
    # IMPORTANT: I have tried to create a matrix created by "loss" that is the same size as input -> wouldnt work as intended as it did not retain_graph, it created bizzare outcome!
    if bs > 1:
        logging.warning("Batchsize more than 1 could result in some inputs don't have gradient!")
    for i in range(bs):
        for j in range(nodes):
            for k in range(latent_dim):
                loss[i,0].backward(x[i,j,k],retain_graph=True)  # backgrad w.r.t each element of x
    if x.grad==None:
        return None
    while (x.grad.sum()==0):   # sometime backward() doesnt work and i dont know why, so i do this
        loss = predictor(x)
        print("this while loop again!")
        if (loss.item()==0 or loss.item()==1):   # this means there is no grad
            break
        for i in range(bs):    
            for j in range(nodes):
                for k in range(latent_dim):
                    loss[i,0].backward(x[i,j,k],retain_graph=True) 
        
    if return_grad:
        return x.grad
    else:
        _x = x.clone()
        while ((predictor(_x)-predictor(x)).sum().item()<bs*epsilon):   # better than og by a certain value
            if lr>min_lr:
                _x = x.clone()
                _x = x.data + lr*x.grad      # gradient ascent
                lr *= lr_decay       # learning rate decay
            else:
                break                           
        return _x
    
'''
This function is the same as the one above, however more compact and efficient, but: I found that 
only when lr > 100 that the returned value _x could noticably change the predictors' outputs.
So I decided to use the first one. (This may not be the best idea, but I tried and it worked so...
maybe I was right, idk)

'''
def grad_ascent2(predictor,x,device,lr=1,lr_decay=0.5,epsilon=1e-2,return_grad=False,min_lr=1):
    predictor.to(device)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad = False
    bs, _, _ = x.shape
    x.to(device)
    out = predictor(x)
    grad = torch.autograd.grad(out,x,torch.ones_like(out))
    grad = grad[0]
    _x = x.clone()
    if return_grad:
        return grad
    else:
        while ((predictor(_x)-predictor(x)).sum().item()<bs*epsilon):   # better than og by a certain value
            if lr>min_lr:
                _x = x.clone()
                _x = x.data + lr*grad      # gradient ascent
                lr *= lr_decay       # learning rate decay
            else:
                break
        return _x 
# NOTE: predictors is a list of predictor
# WARNING: the shape of x is (bs,13,latent_dim), bs SHOULD BE 1, else it may confuses backward()
def grad_ascent_group(predictors,x,device,lr=5e-1,lr_decay=0.5,epsilon=5e-2,min_lr=1e-5):
    num_predictor = len(predictors)   # number of predictors in the list
    # calculate gradient of all predictors
    grad = 0
    for i in range(num_predictor):
        predictors[i].eval()
        for param in predictors[i].parameters():    # this will not perform grad_ascent for the predictor but the input 
            param.requires_grad = False
        grad += grad_ascent(predictors[i],x,device,lr=lr,lr_decay=lr_decay,epsilon=epsilon,return_grad=True,min_lr=min_lr)
    grad *= 1/num_predictor    # normalize
    # initialize temporal "t" to check if the updated version of the input is better
    _x = x.clone()
    t = 0
    while (t<epsilon):    # better than og by a certain value
        if lr>min_lr:
            _x = x.clone()
            _x = x.data + lr*grad      # gradient ascent
            lr *= lr_decay 
            t = 0    
            for i in range(num_predictor):     # recalculate temporal value for the while loop
                t += (predictors[i](_x)-predictors[i](x)).sum().item()
        else: 
            break
    # this will return how much would the model improve (t) and the new value (_x)
    return _x, t

def grad_ascent_group2(predictors,x,device,lr=5e-1,lr_decay=0.5,epsilon=5e-2,min_lr=1e-5):
    num_predictor = len(predictors)   # number of predictors in the list
    # calculate gradient of all predictors
    grad = 0
    for i in range(num_predictor):
        predictors[i].eval()
        for param in predictors[i].parameters():    # this will not perform grad_ascent for the predictor but the input 
            param.requires_grad = False
        grad += grad_ascent2(predictors[i],x,device,lr=lr,lr_decay=lr_decay,epsilon=epsilon,return_grad=True,min_lr=min_lr)
    grad *= 1/num_predictor    # normalize
    _x = x.data + lr*grad      # gradient ascent
    return _x


def infer_mean_std(predictors,x):
    num_predictor = len(predictors)
    # x shape (bs,node_num,latent_dim)
    # mean, shape (bs)
    mean = 0
    outputs = []
    for i in range(num_predictor):
        predictors[i].eval()
        for param in predictors[i].parameters():
            param.requires_grad = False
        temp = predictors[i](x).squeeze()
        mean += temp     # shape (bs)
        outputs.append(temp)
    mean *= 1/num_predictor

    # standard deviation, shape (bs)
    std = 0
    for i in range(num_predictor):
        std += (outputs[i] - mean)**2
    std *= 1/num_predictor
    std = torch.sqrt(std)
    return mean, std

# return the best index andadj according to predictors
def evaluate_adjs_by_predictors(predictors,autoencoder,list_adj,num_return,device,mode="ITS",population=None):
    if autoencoder!= None:
        autoencoder.to(device)
        autoencoder.eval()    # just use autoencoder to project adjs to latent space
        for param in autoencoder.parameters():
            param.requires_grad = False
    for i, adj in enumerate(list_adj):
        list_adj[i] = adj.to(device)
    adjs = torch.stack(list_adj,dim=0)  # shape (bs,4,13,13)
    if autoencoder!= None:
        _, _, adjs = autoencoder(adjs) # shape (bs,2300)
    means, stds = infer_mean_std(predictors,adjs)
    if mode == "mean":
        acquisitions = list(torch.split(means,1,dim=0))  # a list of bs elements
    elif mode == "ITS":
        acquisitions = torch.normal(means,stds)   # shape (bs)
        acquisitions = list(torch.split(acquisitions,1,dim=0))  # a list of bs elements
    elif mode == "EI":
        assert population!=None, "The population must be provided for EI mode"
        label_max = -1
        for label in population.keys():
            if label > label_max:
                label_max = label
        means = means.cpu().detach().numpy()
        stds = stds.cpu().detach().numpy()
        label_max = label_max.clone().cpu().detach().numpy()
        acquisitions = expected_improvement(means,stds,label_max)
        acquisitions = torch.from_numpy(acquisitions).to(device)
        acquisitions = list(torch.split(acquisitions,1,dim=0))
    elif mode == "UCB":
        means = means.cpu().detach().numpy()
        stds = stds.cpu().detach().numpy()
        acquisitions = upper_confidence_bound(means,stds)
        acquisitions = list(torch.split(acquisitions,1,dim=0))
    else:
        raise ValueError("mode must be one of the following: mean, ITS, EI, UCB")
    max = 0
    list_index_max = []
    list_acqui_max = []
    for _ in range(num_return):
        for i, acqui in enumerate(acquisitions):
            if acqui>max:
                index_max = i
                acqui_max = acqui
                max = acqui
        list_index_max.append(index_max)
        list_acqui_max.append(acqui_max)
        max = 0
        # so that we dont have to add this index anymore
        acquisitions[index_max] = -1
    return list_index_max, list_acqui_max

def expected_improvement(mean, std_dev, f_best):
    z = (mean - f_best) / std_dev
    ei = std_dev * (z * norm.cdf(z) + norm.pdf(z))
    return ei

def upper_confidence_bound(mean, std_dev, kappa=0.5):
    ucb = mean + kappa * std_dev
    return ucb

# predictors is a list of n predictors, same for list_adj, autoencoder is Model_VAE
# this function will take output of n-1 to cross validate the remaining predictor
def cross_validation_label(list_adj,predictors,autoencoder,device):  
    autoencoder.to(device)
    autoencoder.eval()   # disable dropout and self.training = False (see VAE_for_3d_graph.py)
    for param in autoencoder.parameters():
        param.requires_grad = False
    num_predictor = len(predictors)
    list_adj = torch.stack(list_adj,dim=0)  # shape (bs,4,13,13)
    list_adj = list_adj.to(device)
    _, mu, _ = autoencoder(list_adj)    # shape (bs,node_num,latent_dim)
    outputs = []
    for i in range(num_predictor):
        predictors[i].eval()
        predictors[i].to(device)
        for param in predictors[i].parameters():
            param.requires_grad = False
        outputs.append(torch.squeeze(predictors[i](mu)))   # shape: a list of shape (bs)

    # labels[i] contains the mean output of all predictors except predictors[i]
    labels = []
    temp_mean = 0
    for i in range(1,num_predictor):
        temp_mean += outputs[i]/(num_predictor-1)     # this would create the first element of labels 
    # IMPORTANT: this prevents python from passing the address of temp_mean as we wont initialize temp_mean again!
    labels.append(temp_mean.clone())      
    for i in range(1,num_predictor): 
        temp_mean -= outputs[i]/(num_predictor-1)   # remove the outputs of the ith predictor (i in range(1,..) means ith predictors)
        temp_mean += outputs[i-1]/(num_predictor-1)   # ultimately create the cross validation labels of ith predictor
        labels.append(temp_mean.clone())
    return labels    # a list of tensors shape (bs)


# p = [Predictor(16,8) for _ in range(3)]
# p = Predictor(16,8)
# autoencoder = vae.Model_VAE()
# lr = 5e-1
# x = torch.rand((1,13,16),requires_grad=True)  
# _x, t =grad_ascent_group(p,x)
# print(t)
# mean, std = infer_mean_std(p,x)
# print(mean)
# print(std)

# y = grad_ascent2(p,x,"cpu")

# print(p(y)-p(x))

# list_adj = []
# for i in range(4):
#     list_adj.append(util.preprocess_adj(util.randomly_generate())[0])

# autoencoder = vae.Model_VAE()
# labels = cross_validation_label(list_adj,p,autoencoder)