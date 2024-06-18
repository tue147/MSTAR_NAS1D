import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb as wb

from VAE_for_3d_graph import *
from util import *
from predictor import *
from training_controller import *


EPOCHS = 50
INITIALIZE_NUM = 400
number_of_nodes = 13
mutate_num = 10  # number of mutation to initialize

# predictors
batch_size_pred = 16 
channel = 2300
lr_pred = 1e-3
epochs_pred = 10

# training model
batch_size = 128
epochs = 18
lr = 1e-2
T_max = 16
max_count = 3

# model 
num_cell = 6
channel_at_cell = 128
out_dim = 5
dropout = 0.0
grad_clip = None
ic = 12
scheduler_type = torch.optim.lr_scheduler.OneCycleLR
optimizer_type = torch.optim.Adam
criterion_type = nn.functional.binary_cross_entropy
loss = nn.functional.binary_cross_entropy
pool = AdaptiveConcatPool1d
weight_decay = 1e-2
conv_ker = [1,3,5,9,19,39]
avg_max_ker = [3,5,9]
bottleneck_reduction = 4
num_classifier_nodes = channel_at_cell*2
stride = None
skip = False

# others
mutate_prob = 0.8
crossover_prob = 0.0
dom_prob = 0.7
sub_prob = 0.3
pruning_prob = 0.2
cross_mutate_prob = 0.08
num_predictors = 10
num_samples = 4 # 64
num_outcome = 24 # 128, or else must greater than 1
num_return = 4
mode = "EI"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# run = wb.init(project="1D_NAS", entity="aiotlab")
# config = wb.config
# config.lr_pred = lr_pred
# config.EPOCHS = EPOCHS
# config.epochs_pred = epochs_pred
# config.autoencoder = True
# config.epochs_ae = epochs_ae
# config.mode = mode
# config.type = 81
run = None


# add data
def load_data(num_class):
    if num_class==5:
        ytrain = np.load("y_super_train.npy")
        yval = np.load("y_super_val.npy")
        ytest = np.load("y_super_test.npy")
    elif num_class==23:
        ytrain = np.load("y_sub_train.npy")
        yval = np.load("y_sub_val.npy")
        ytest = np.load("y_sub_test.npy")
    elif num_class==71:
        ytrain = np.load("y_all_train.npy")
        yval = np.load("y_all_val.npy")
        ytest = np.load("y_all_test.npy")
    elif num_class==19:
        ytrain = np.load("y_form_train.npy")
        yval = np.load("y_form_val.npy")
        ytest = np.load("y_form_test.npy")
    elif num_class==44:
        ytrain = np.load("y_diag_train.npy")
        yval = np.load("y_diag_val.npy")
        ytest = np.load("y_diag_test.npy")
    elif num_class==12:
        ytrain = np.load("y_rhythm_train.npy")
        yval = np.load("y_rhythm_val.npy")
        ytest = np.load("y_rhythm_test.npy")

    ytrain = torch.from_numpy(ytrain)
    yval = torch.from_numpy(yval)
    ytest = torch.from_numpy(ytest)

    if num_class==71:
        xtrain = np.load("X_train_all.npy")
        xval = np.load("X_val_all.npy")
        xtest = np.load("X_test_all.npy")

    elif num_class==5 or num_class==23 or num_class==44:
        xtrain = np.load("X_train_bandpass.npy",mmap_mode="r+")
        xval = np.load("X_val_bandpass.npy",mmap_mode="r+")
        xtest = np.load("X_test_bandpass.npy",mmap_mode="r+")
    elif num_class==19:
        xtrain = np.load("X_train_form.npy",mmap_mode="r+")
        xval = np.load("X_val_form.npy",mmap_mode="r+")
        xtest = np.load("X_test_form.npy",mmap_mode="r+")
    elif num_class==12:
        xtrain = np.load("X_train_rhythm.npy",mmap_mode="r+")
        xval = np.load("X_val_rhythm.npy",mmap_mode="r+")
        xtest = np.load("X_test_rhythm.npy",mmap_mode="r+")

    xtrain = torch.from_numpy(xtrain)
    xval = torch.from_numpy(xval)
    xtest = torch.from_numpy(xtest)

    # xtrain = torch.concatenate([xtrain,xval],dim=0)
    # ytrain = torch.concatenate([ytrain,yval],dim=0)

    # dataloader
    train_data = (xtrain,ytrain)
    val_data = (xval,yval)
    test_data = (xtest,ytest)
    # return train_data, val_data, test_data
    return train_data, val_data
# add sota architectures
list_sota_adj = []
shape = (number_of_nodes,number_of_nodes)

def create_adj_mannually(conv_mat,max_mat,SE_mat,ide_mat):
    return torch.stack([conv_mat,max_mat,SE_mat,ide_mat])    # stack them 
def create_sparse_mat(index=None,value=None):
    if index != None and value != None:
        return torch.sparse_coo_tensor(index,value,shape).to_dense()    # create a sparse matrix type coo
    else:
        return torch.zeros(shape)   # create zeros 
# SOTA our model
def Our_baseline():
    index = [[0,4,4,4,3,0],     # conv
            [1,8,9,10,7,12]]
    value = [1,9,19,39,3,3]
    conv_mat = create_sparse_mat(index,value)
    index = [[0],               # max
            [3]]
    value = [3]
    max_mat = create_sparse_mat(index,value)
    avg_mat = create_sparse_mat()
    index = [[1],               # max
            [4]]
    value = [1]
    ide_mat = create_sparse_mat(index,value)
    return create_adj_mannually(conv_mat,max_mat,avg_mat,ide_mat)
def InceptionTime():
    index = [[0,1,1,1,3],     # conv
            [1,4,5,6,7]]
    value = [1,9,19,39,1]
    conv_mat = create_sparse_mat(index,value)
    index = [[0],               # max
            [3]]
    value = [3]
    max_mat = create_sparse_mat(index,value)
    SE_mat = create_sparse_mat()
    index = [[0],               # identity
            [13]]
    value = [1]
    ide_mat = create_sparse_mat(index,value)
    return create_adj_mannually(conv_mat,max_mat,SE_mat,ide_mat)

list_sota_adj.append(Our_baseline())
list_sota_adj.append(InceptionTime())
# list_sota_adj = None

# Initialize everything
data = load_data(num_class=out_dim) 
# data = ((torch.rand((40,12,1000)),torch.tensor(np.random.choice(2,(40,5),p=[0.2,0.8]))),(torch.rand((40,12,1000)),torch.tensor(np.random.choice(2,(40,5),p=[0.2,0.8]))))

predictors = []
for _ in range(num_predictors):
    predictors.append(Predictor_linear(channel=channel).to(device))
    
population = initialize_population_save_each_model(INITIALIZE_NUM,data,device,
                        list_adj_to_mutate=list_sota_adj,mutate_num=mutate_num,pruning_prob=pruning_prob,
                        lr=lr,epochs=epochs,batchsize=batch_size,max_count=max_count,
                        num_cells=num_cell,channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,
                        scheduler_type=scheduler_type,T_max=T_max,optimizer_type=optimizer_type,ic=ic,
                        criterion_type=criterion_type,loss=loss,weight_decay=weight_decay,conv_ker=conv_ker,
                        avg_max_ker=avg_max_ker,skip=skip,num_classifier_nodes=num_classifier_nodes,
                        bottleneck_reduction=bottleneck_reduction,pool=pool,num_node=number_of_nodes,
                        wandb=run,grad_clip=grad_clip,stride=stride)

# # load Initialized population
# population = torch.load("Population_initialized_val_SE",map_location=device)
autoencoder = torch.load("autoencoder800",map_location=device)

# train_predictors(predictors,autoencoder,population,device,wandb=run,batch_size=batch_size_pred,lr=lr_pred,epochs=100,)

# the main loop
for EPOCH in range(EPOCHS):
    # train the autoencoder and predictors with the current population
    run.log({"EPOCH":EPOCH})
    train_predictors(predictors,autoencoder,population,device,wandb=run,batch_size=batch_size_pred,lr=lr_pred,epochs=epochs_pred,)
    # create a sample_list then generate a population of num_outcome models to self train the predictors
    sample_list = return_best_adjs(population,num_samples)
    sample_list = generate_for_searching_phase(sample_list,num_outcome,mutate_prob=mutate_prob,crossover_prob=crossover_prob,
                                            dom_prob=dom_prob,sub_prob=sub_prob,pruning_prob=pruning_prob,cross_mutate_prob=cross_mutate_prob,
                                            num_node=number_of_nodes,conv_ker=conv_ker,avg_max_ker=avg_max_ker)
    
    indexes, means = evaluate_adjs_by_predictors(predictors,autoencoder,sample_list,num_return=num_return,device=device,mode=mode,population=population)
    for i, index in enumerate(indexes):
        label = train_nn(data,sample_list[i],device,lr=lr,epochs=epochs,batchsize=batch_size,num_cells=num_cell,ic=ic,
                        channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,stride=stride,
                        num_node=number_of_nodes,pool=pool,bottleneck_reduction=bottleneck_reduction,
                        conv_ker=conv_ker,avg_max_ker=avg_max_ker,skip=skip,num_classifier_nodes=num_classifier_nodes,
                        optimizer_type=optimizer_type,scheduler_type=scheduler_type,T_max=T_max,max_count=max_count,
                        grad_clip=grad_clip,loss=loss,weight_decay=weight_decay,criterion_type=criterion_type)
        print("The different between prediction and label is:")
        print(abs(scale(label)-means[i]))
        run.log({"diff":abs(scale(label)-means[i])})
        run.log({"label":label})
        adj, _, _ = preprocess_adj(sample_list[index].to(device)) if not util._SE_ else preprocess_adj_SE(sample_list[index].to(device))
        population[torch.tensor([scale(label)]).to(device)] = adj.to(device)

    torch.save(EPOCH,"EPOCH_SE_avg")
    torch.save(population,"Population_SE_avg")
    # torch.save(autoencoder,"Autoencoder4")
    torch.save(predictors, "Predictors_SE_avg")


# find the best adj in the population
num_adj = len(population)
list_adj = list(population.keys())
max_ = list_adj[0].item()
best = population[list_adj[0]]
for i in range(num_adj):
    if max_ < list_adj[i].item():
        max_ = list_adj[i].item()
        best = population[list_adj[i]]

print("TRAINING_NAS FINISHED!\n=============================================================")
print("BEST SCORE: {}".format(max_))
    
# save everything
torch.save(EPOCH,"EPOCH_SE_avg")
torch.save(predictors, "Predictors_SE_avg")
# torch.save(autoencoder,"Autoencoder4")
torch.save(population,"Population_SE_avg")
torch.save(best,"Best_adj_SE_avg")