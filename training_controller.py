import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import lightning.pytorch as pl 
from pytorch_lightning.utilities import rank_zero_info
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging, LearningRateFinder

from VAE_for_3d_graph import *
from util import *
from predictor import *

# NOTE: when appending the population new elements, makesure append clone() of that data
# IMPORTANT: DO NOT change the order of population dictionary while training

def train_autoencoder(autoencoder,population,device,wandb,
                        batch_size=16,lr=1e-3,epochs=5,T_max=0,
                        gamma=0.5,max_count=3,loss_adj=nn.MSELoss,
                        optimizer_type=torch.optim.Adam,
                        scheduler = torch.optim.lr_scheduler.OneCycleLR
                        ):
    '''
    population is a dict that values are tensors that have the shape of (4,13,13) and keys are labels 
    (or performances of those models). The values must be preprocessed matrix (see preprocess_adj() in util.py)

    '''
    population_train, population_val = population
    list_adj_train = list(population_train.values())    # list of torch.tensor shape (4,13,13)
    list_adj_train = torch.stack(list_adj_train,dim=0)  # shape (len(list_adj),4,13,13)

    list_adj_val = list(population_val.values())   
    list_adj_val = torch.stack(list_adj_val,dim=0)
    
    # criterion = VAEReconstructed_L2_Loss(gamma = gamma, loss_adj=loss_adj)
    criterion = nn.MSELoss()
    optimizer = optimizer_type(autoencoder.parameters(),lr=lr)
    train_data = DataLoader(TensorDataset(list_adj_train), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(TensorDataset(list_adj_val), batch_size=batch_size, shuffle=False)
    scheduler = scheduler(optimizer,lr,steps_per_epoch=len(train_data),epochs=T_max)
    
    # switching on for training
    autoencoder.to(device)
    for param in autoencoder.parameters():
        param.requires_grad = True

    count = 0
    for epoch in tqdm(range(epochs)):
        autoencoder.train()
        print("\nAutoencoder training:")
        print("Epoch {}:\n>----------------------------------------".format(epoch+1))
        loss_per_epoch = [] 
        for adjs in tqdm(train_data):
            adjs = adjs[0]      # adjs is a list of 1 element    
            adjs = adjs.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            adjs_recon,_,_ = autoencoder(adjs)   # mu: (bs,num_node,latent_dim), adj_recon: (bs,4,13,13) but in sigmoid form (which is what we want!)
            # adjs_recon_sigmoid = torch.split(adjs_recon_sigmoid,1,dim=0)
            # adjs_recon = []
            # for adj_recon_sigmoid in adjs_recon_sigmoid:
            #     adjs_recon.append(preprocess_adj(adj_recon_sigmoid.squeeze())[0]) 
            # adjs_recon = torch.stack(adjs_recon,dim=0)
            # loss = criterion(adjs_recon, mu, logvar, adjs)
            loss = criterion(adjs_recon, adjs)
            if wandb!=None:
                wandb.log({"AE_loss":loss,
                            })
            loss.backward()
            optimizer.step()
            if epoch < T_max:
                scheduler.step()
            loss_per_epoch.append(loss.item())
        print("\tTraining loss:{:.4f}".format(sum(loss_per_epoch)/len(loss_per_epoch)))
        print("\tFinished epoch {}\n>----------------------------------------".format(epoch+1))

        with torch.no_grad():
            autoencoder.eval()
            all_loss_val = []
            for adjs in tqdm(val_data):
                adjs = adjs[0]      # adjs is a list of 1 element    
                adjs = adjs.to(device)

                # forward + backward + optimize
                adjs_recon, _,_ = autoencoder(adjs)
                loss = criterion(adjs_recon, adjs)
                # loss = criterion(adjs_recon, mu, logvar, adjs)

                all_loss_val.append(loss.item())

            # if overfit then stop
            if epoch >= T_max:       
                if all_loss_val[-1] < all_loss_val[-2]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
            print("epoch {}: Validation loss:{:.4f}".format(epoch+1, sum(all_loss_val)/len(all_loss_val)))

def train_autoencoder1(autoencoder,population,device,wandb,
                        batch_size=16,lr=1e-3,epochs=5,T_max=0,
                        gamma=0.5,max_count=3,loss_adj=nn.MSELoss,
                        optimizer_type=torch.optim.Adam,
                        scheduler = torch.optim.lr_scheduler.OneCycleLR
                        ):
    '''
    population is a dict that values are tensors that have the shape of (4,13,13) and keys are labels 
    (or performances of those models). The values must be preprocessed matrix (see preprocess_adj() in util.py)

    '''
    population_train, population_val = population
    list_adj_train = list(population_train.values())    # list of torch.tensor shape (4,13,13)
    list_adj_train = torch.stack(list_adj_train,dim=0)  # shape (len(list_adj),4,13,13)

    list_adj_val = list(population_val.values())   
    list_adj_val = torch.stack(list_adj_val,dim=0)
    
    criterion = VAEReconstructed_L2_Loss(gamma = gamma, loss_adj=loss_adj)
    # criterion = nn.MSELoss()
    optimizer = optimizer_type(autoencoder.parameters(),lr=lr)
    train_data = DataLoader(TensorDataset(list_adj_train), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(TensorDataset(list_adj_val), batch_size=batch_size, shuffle=False)
    scheduler = scheduler(optimizer,lr,steps_per_epoch=len(train_data),epochs=T_max)
    
    # switching on for training
    autoencoder.to(device)
    for param in autoencoder.parameters():
        param.requires_grad = True

    count = 0
    for epoch in tqdm(range(epochs)):
        autoencoder.train()
        print("\nAutoencoder training:")
        print("Epoch {}:\n>----------------------------------------".format(epoch+1))
        loss_per_epoch = [] 
        for adjs in tqdm(train_data):
            adjs = adjs[0]      # adjs is a list of 1 element    
            adjs = adjs.to(device)
            optimizer.zero_grad()
            # forward + backward + optimize
            adjs_recon,mu,logvar = autoencoder(adjs)   # mu: (bs,num_node,latent_dim), adj_recon: (bs,4,13,13) but in sigmoid form (which is what we want!)
            loss = criterion(adjs_recon, mu, logvar,adjs)
            if wandb!=None:
                wandb.log({"AE_loss":loss,
                            })
            loss.backward()
            optimizer.step()
            if epoch < T_max:
                scheduler.step()
            loss_per_epoch.append(loss.item())
        print("\tTraining loss:{:.4f}".format(sum(loss_per_epoch)/len(loss_per_epoch)))
        print("\tFinished epoch {}\n>----------------------------------------".format(epoch+1))

        with torch.no_grad():
            autoencoder.eval()
            all_loss_val = []
            for adjs in tqdm(val_data):
                adjs = adjs[0]      # adjs is a list of 1 element    
                adjs = adjs.to(device)

                # forward + backward + optimize
                adjs_recon,mu,logvar = autoencoder(adjs)   
                loss = criterion(adjs_recon, mu, logvar,adjs)
                # loss = criterion(adjs_recon, mu, logvar, adjs)

                all_loss_val.append(loss.item())

            # if overfit then stop
            if epoch >= T_max:       
                if all_loss_val[-1] < all_loss_val[-2]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
            print("epoch {}: Validation loss:{:.4f}".format(epoch+1, sum(all_loss_val)/len(all_loss_val)))

def train_predictors(predictors,autoencoder,population,device,wandb,
                    batch_size=16,lr=1e-3,epochs=1,
                    loss_func=nn.MSELoss,optimizer_type=torch.optim.Adam,
                    ):
    '''
    predictors is a list of Predictors, population is a dict that values are tensors that have the 
    shape of (4,13,13) and keys are labels (or performances of those models)

    '''
    list_adj = list(population.values())    # list of torch.tensor shape (4,13,13)
    list_adj = torch.stack(list_adj,dim=0)  # shape (len(list_adj),4,13,13)
    list_labels = list(population.keys())   # list of labels shape (1)
    list_labels = torch.stack(list_labels,dim=0)   # shape (len(list_labels),1)
    train_data = DataLoader(TensorDataset(list_adj,list_labels), batch_size=batch_size, shuffle=True)
    
    # freeze autoencoder
    if autoencoder!=None:
        autoencoder.to(device)
        autoencoder.eval()    # just use autoencoder to project adjs to latent space
        for param in autoencoder.parameters():
            param.requires_grad = False
    

    # training loop, we will train each predictor one by one
    num_predictor = len(predictors)
    for i in tqdm(range(num_predictor)): 
        print("\nINITIALIZE PREDICTORS TRAINING:>=====================================")   
        # turn on require_grad 
        predictors[i].train().float()
        predictors[i].to(device)
        for param in predictors[i].parameters():
            param.requires_grad = True
        
        criterion = loss_func()
        optimizer = optimizer_type(predictors[i].parameters(),lr=lr,weight_decay=1e-5)
        

        for epoch in range(epochs):
            if num_predictor==1:
                print("\nPREDICTOR training:")
            else:
                print("\nPREDICTOR{} training:".format(i+1))
            print("Epoch {}:\n>----------------------------------------".format(epoch+1))
            loss_per_epoch = [] 
            for adjs,labels in train_data:    
                adjs = adjs.to(device).float() 
                labels = labels.to(device).float()
                optimizer.zero_grad()

                # forward + backward + optimize
                # IMPORTANT, I have to compute mu all over agrain bc backward() requires retain_graph=True if backward() a tensor more than 1
                if autoencoder!=None:
                    _, _, repre = autoencoder(adjs)   # mu: (bs,num_node,latent_dim)
                    outputs = predictors[i](repre)    # outputs: (bs,1)
                else:
                    outputs = predictors[i](adjs)
                loss = criterion(outputs,labels.float())
                if wandb!=None:
                    if num_predictor==1:
                        wandb.log({"Predictor_loss_cross_valid":loss,
                                    })
                    else:
                        wandb.log({"Predictor_loss":loss,
                                    })
                loss.backward()
                optimizer.step()
                loss_per_epoch.append(loss.detach().item())
            print("\tTraining loss:{:.4f}".format(sum(loss_per_epoch)/len(loss_per_epoch)))
            print("\tFinished epoch {}\n>----------------------------------------".format(epoch+1))


def self_train_predictors(predictors,autoencoder,list_adj,device,num_matrix_in_population,wandb,
                        logvar0=1e-2,batch_size=4,lr=1e-4,epochs=1,
                        loss_func=nn.MSELoss,optimizer_type=torch.optim.Adam,
                        mutate_prob=0.5,pruning_prob=0.2,num_node=13
                        ):
    '''
    this function takes a list_adj to generate a list of new adj then self train predictors on that list, 
    and then return the mu and logvar of all newly generated model

    NOTE: num_matrix_in_population is the number of adjs in the list_new_adj, or the parameter: num_outcome (see
    generate_for_self_train_predictor() )

    '''
    num_predictors = len(predictors)
    # list_new_adj is a list unpreprocessedd adj
    list_new_adj = generate_for_searching_phase(list_adj,num_matrix_in_population,mutate_prob=mutate_prob,pruning_prob=pruning_prob,num_node=num_node)
    for i in range(len(list_new_adj)):
        list_new_adj[i] = list_new_adj[i].to(device)   # add to device to train
    # the cross_validation_label will not change list_new_adj
    labels = cross_validation_label(list_new_adj,predictors,autoencoder,device)   # this function doesnt compute grad but needs to put predictors into eval() mode
    temp_population = {}
    # labels[i] is the cross validation labels of all other predictors except ith predictors
    for i in range(num_predictors):
        temp_list_labels = torch.split(labels[i],1,dim=0)    # return a list of tensors shape (1)
        for j,label in enumerate(temp_list_labels):
            temp_population[label] = list_new_adj[j]  # there is no need to put to device, bc when calling add_to_population, It will put all adjs to device
        train_predictors([predictors[i]],autoencoder,temp_population,device,wandb,
                        batch_size=batch_size,lr=lr,epochs=epochs,
                        loss_func=loss_func,optimizer_type=optimizer_type)
    # append adjs that predictors agree upon (logvar < logvar0)
    list_new_adj_ = torch.stack(list_new_adj,dim=0)  # shape (num_matrix_in_population,4,13,13)
    autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    list_new_adj_ = list_new_adj_.to(device)
    _, mu, _ = autoencoder(list_new_adj_)    
    mean, std = infer_mean_std(predictors,mu)  # shape (num_matrix_in_population)
    check = std < logvar0                      # check has the shape same as std and contains only True or False
    return_adj = []      
    return_mean = []
    for i in range(num_matrix_in_population):
        if check[i]:
            return_adj.append(list_new_adj[i])    # append adj
            # append the predicted performance, mean[i] is a tensor shape (), so that we know if the models were assessed and added to population by the predictors
            return_mean.append(torch.tensor([mean[i].item()]))  
    return return_adj, return_mean


# generate ground truth by training from scratch
def train_nn(data,adj,device,lr=1e-2,epochs=20,batchsize=128,num_cells=6,ic=12,channel_at_cell=128,out_dim=5,
            dropout=0.0,stride = None,num_node=13,pool=AdaptiveConcatPool1d,bottleneck_reduction=4,
            conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],skip=False,num_classifier_nodes=128*2,
            optimizer_type=torch.optim.AdamW,scheduler_type=torch.optim.lr_scheduler.OneCycleLR,
            T_max=16,max_count=3,grad_clip=None,loss=nn.functional.binary_cross_entropy,weight_decay=1e-2,
            criterion_type=nn.functional.binary_cross_entropy,**pool_kwargs
            ):
    '''
    data comprise:  train, val, 
    adj: is the encoding or a nn and has been preprocessed,
    max_count: counts the number of consecutive times that current val_auc < previous val_auc
    val_auc_all: stores all validation auc
    -> this function will return max(val_auc_all)
    '''
    net = Create_model(adj.clone(),num_cells=num_cells,channel_at_cell=channel_at_cell,ic=ic,
                        bottleneck_reduction=bottleneck_reduction,out_dim=out_dim,dropout=dropout,
                        num_node=num_node,stride=stride,pool=pool,skip=skip,
                        conv_ker=conv_ker,avg_max_ker=avg_max_ker,loss=loss,
                        num_classifier_nodes=num_classifier_nodes,**pool_kwargs
                        ).to(device)
    # load data
    traindata, valdata = data
    xtrain, ytrain = traindata
    xval, yval = valdata
    train_data = DataLoader(TensorDataset(xtrain,ytrain), batch_size=batchsize, shuffle=True)
    val_data = DataLoader(TensorDataset(xval,yval), batch_size=batchsize, shuffle=False)

    # optimizer, scheduler, criterion
    optimizer = optimizer_type(net.parameters(),lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_type(optimizer,lr,steps_per_epoch=len(train_data),epochs=T_max,verbose=False)
    criterion = criterion_type

    val_auc_all = []
    count = 0
    for epoch in range(epochs):
        net.train()
        print("\nTRAINING MODEL:===========================================================")
        print("\nEpoch {}:\n>-----------------------------------------------------------".format(epoch+1)) 
        all_outs_train = []
        all_labels_train = []
        all_loss_train = []
        for inputs,labels in train_data:
            inputs,labels = inputs.to(device).float(),labels.to(device).float()
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            if grad_clip != None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            # for onecyclelr
            if epoch < T_max:
                scheduler.step()

            all_outs_train.append(outputs.cpu().detach().numpy())
            all_labels_train.append(labels.cpu().detach().numpy())
            all_loss_train.append(loss)
        all_outs_train = np.concatenate(all_outs_train,axis = 0)
        all_labels_train = np.concatenate(all_labels_train,axis = 0)
        print("Epoch {}: Training loss:{:.4f}".format(epoch+1, sum(all_loss_train)/len(all_loss_train)))
        print("Training AUC score of epoch {}:{}".format(epoch+1,roc_auc_score(all_labels_train,all_outs_train,average = "macro")))
        
        with torch.no_grad():
            net.eval()
            all_outs_val = []
            all_labels_val = []
            all_loss_val = []
            for inputs,labels in val_data:
                inputs,labels = inputs.to(device).float(),labels.to(device).float()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels.float())

                all_outs_val.append(outputs.cpu().detach().numpy())
                all_labels_val.append(labels.cpu().detach().numpy())
                all_loss_val.append(loss)
            all_outs_val = np.concatenate(all_outs_val,axis = 0)
            all_labels_val = np.concatenate(all_labels_val,axis = 0)
            val_auc_all.append(roc_auc_score(all_labels_val,all_outs_val,average = "macro"))

            # if overfit then stop
            if epoch >= T_max:       
                if val_auc_all[-1] < val_auc_all[-2]:
                    count += 1
                    if count >= max_count:
                        break
                else:
                    count = 0
            print("================ Validation epoch:{}=================".format(epoch+1))
            print("Validation loss:{:.4f}".format(sum(all_loss_val)/len(all_loss_val)))
            print("Validation AUC score of epoch {}:{}".format(epoch+1,val_auc_all[-1]))
    print("\nfinished training =================================================")
    m = max(val_auc_all)
    print("Best validation performance: {}".format(m))

    return m   # return the average recorded

def train_nn2(data,adj,device,lr=1e-2,epochs=20,batchsize=128,num_cells=6,ic=12,channel_at_cell=128,out_dim=5,
                dropout=0.0,stride = None,num_node=13,pool=AdaptiveConcatPool1d,bottleneck_reduction=4,
                conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],skip=False,num_classifier_nodes=128*2,
                loss=nn.functional.binary_cross_entropy,**pool_kwargs):
    
    net = Create_model(adj.clone(),num_cells=num_cells,channel_at_cell=channel_at_cell,ic=ic,
                        bottleneck_reduction=bottleneck_reduction,out_dim=out_dim,dropout=dropout,
                        num_node=num_node,stride=stride,pool=pool,skip=skip,
                        conv_ker=conv_ker,avg_max_ker=avg_max_ker,loss=loss,
                        num_classifier_nodes=num_classifier_nodes,**pool_kwargs
                        ).to(device)
    # load data
    traindata, valdata = data
    xtrain, ytrain = traindata
    xval, yval = valdata
    train_data = DataLoader(TensorDataset(xtrain,ytrain), batch_size=batchsize, shuffle=True)
    val_data = DataLoader(TensorDataset(xval,yval), batch_size=batchsize, shuffle=False)
    class LoggingCallback(pl.Callback):

        def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            rank_zero_info("***** Test results *****")
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    rank_zero_info("{} = {}\n".format(key, str(metrics[key])))
                
    class FineTuneLearningRateFinder(LearningRateFinder):
        def __init__(self, milestones, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.milestones = milestones

        def on_fit_start(self, *args, **kwargs):
            return

        def on_train_epoch_start(self, trainer, pl_module, *args, **kwargs):
            if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
                self.lr_find(trainer, pl_module, *args, **kwargs)

    trainer = pl.Trainer(max_epochs=epochs,callbacks=[LoggingCallback(), 
                                                    StochasticWeightAveraging(0.5 * lr,swa_epoch_start=0.75), #0.5e-3
                                                    # FineTuneLearningRateFinder(milestones=[], max_lr=1e-2, min_lr=1e-8, attr_name='lr')
                                                    ])
    trainer.fit(model=net, train_dataloaders=train_data, val_dataloaders=val_data)
    results = trainer.test(model=net, dataloaders=val_data)
    print(results)
    print("\nfinished training =================================================")
    print("Recorded validation performance: {}".format(results[0]['test_acc_step']))
    return results[0]['test_acc_step']
# add adjs and labels to the population
def add_to_population(population,list_adj,list_labels,device,need_scale=True):
    if need_scale:
        for i in range(len(list_labels)):
            population[scale(list_labels[i].to(device))] = list_adj[i].to(device)   # the labels and adjs are all in device 
    else:
        for i in range(len(list_labels)):
            population[list_labels[i].to(device)] = list_adj[i].to(device)   # the labels and adjs are all in device 
    return population

def initialize_population(initialize_num,data,device,
                        list_adj_to_mutate=None,mutate_num=0,pruning_prob=0.2,
                        lr=1e-2,epochs=20,batchsize=128,max_count=3,
                        num_cells=6,channel_at_cell=128,out_dim=5,dropout=0.0,
                        scheduler_type=torch.optim.lr_scheduler.OneCycleLR,T_max=16,
                        optimizer_type=torch.optim.AdamW,
                        criterion_type=nn.functional.binary_cross_entropy,
                        num_node=13,wandb=None
                        ):
    number_of_adjs_need_to_add_left = initialize_num    # count how many adjs needs to initialize left
    population = {}
    if list_adj_to_mutate != None:
        num_mutate_adjs = len(list_adj_to_mutate)
        assert mutate_num != 0, "Must provide the number of mutation!"
        assert initialize_num>=num_mutate_adjs*mutate_num, "Number of initialize adjs cannot be smaller than the number of mutations!"
    
        for i in range(num_mutate_adjs):   # loop through all list_adj_to_mutate
            mutation_adj = []
            mutation_labels = []
            for _ in range(mutate_num):  
                # this func create a clone of list_adj_to_mutate[i] so it would not change the original 
                # preprocessed_adj, _, _ = preprocess_adj(list_adj_to_mutate[i])
                # completely randomize mutated adj
                # new_adj = reverse_to_sigmoid_random(preprocess_adj(mutate(preprocessed_adj,num_node=num_node))[0])    
                new_adj = mutate(list_adj_to_mutate[i])
                try:
                    label = train_nn(data,new_adj,device,lr,epochs,batchsize,max_count,    # train new_adj
                                    num_cells,channel_at_cell,out_dim,dropout,
                                    scheduler_type,T_max,optimizer_type,criterion_type,num_node=num_node)
                    if wandb is not None:
                        wandb.log({"label":label})
                # I comment the following line because population contains unpreprocessed adjs
                except:
                    print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                    continue
                new_adj, _, _ = preprocess_adj(new_adj)
                mutation_adj.append(new_adj)
                mutation_labels.append(torch.tensor([label]))  # the population's labels only take the form of tensors
            add_to_population(population,mutation_adj,mutation_labels,device)
        number_of_adjs_need_to_add_left -= num_mutate_adjs*mutate_num  # the remaining number of adjs that need to be added
    else:
        assert mutate_num==0, "There is no adjacency matrix to mutate!, please set mutate_num = 0"
    # now for the randomly generating part    
    random_adj = []
    labels_for_random_adj = []
    for _ in range(number_of_adjs_need_to_add_left):
        new_adj = randomly_generate(pruning_prob,num_node=num_node)    
        try:   
            label = train_nn(data,new_adj,device,lr,epochs,batchsize,max_count,    # train new_adj
                            num_cells,channel_at_cell,out_dim,dropout,
                            scheduler_type,T_max,optimizer_type,criterion_type,num_node=num_node)
            if wandb is not None:
                wandb.log({"label":label})
        except:
                print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                continue
        new_adj, _, _ = preprocess_adj(new_adj)
        random_adj.append(new_adj)
        labels_for_random_adj.append(torch.tensor([label]))
    add_to_population(population,random_adj,labels_for_random_adj,device)
    return population

def initialize_population_with_lightning(initialize_num,data,device,wandb=None,
                        list_adj_to_mutate=None,mutate_num=0,pruning_prob=0.2,
                        lr=1e-2,epochs=20,batchsize=128,ic=12,
                        num_cells=6,channel_at_cell=128,out_dim=5,dropout=0.0,
                        stride = None,num_node=13,pool=AdaptiveConcatPool1d,
                        bottleneck_reduction=4,conv_ker=[1,3,5,9,19,39],
                        avg_max_ker=[3,5,9],skip=False,num_classifier_nodes=128*2,
                        loss=nn.functional.binary_cross_entropy,**pool_kwargs
                        ):
    number_of_adjs_need_to_add_left = initialize_num    # count how many adjs needs to initialize left
    # population = torch.load("Population_initialized_SE_test",map_location=device)   # load previous run
    population = {}   
    if list_adj_to_mutate != None:
        num_mutate_adjs = len(list_adj_to_mutate)
        assert mutate_num != 0, "Must provide the number of mutation!"
        assert initialize_num>=num_mutate_adjs*mutate_num, "Number of initialize adjs cannot be smaller than the number of mutations!"
    
        for i in range(num_mutate_adjs):   # loop through all list_adj_to_mutate
            for _ in range(mutate_num):  
                mutation_adj = []
                mutation_labels = []
                # this func create a clone of list_adj_to_mutate[i] so it would not change the original 
                new_adj = mutate(list_adj_to_mutate[i])
                # try:
                label = train_nn(data,new_adj,device,lr=lr,epochs=epochs,batchsize=batchsize,num_cells=num_cells,ic=ic,
                                channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,stride=stride,
                                num_node=num_node,pool=pool,bottleneck_reduction=bottleneck_reduction,conv_ker=conv_ker,
                                avg_max_ker=avg_max_ker,skip=skip,num_classifier_nodes=num_classifier_nodes,
                                loss=loss,**pool_kwargs)
                if wandb is not None:
                    wandb.log({"init_label":label})
                # except:
                #     print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                #     continue
                new_adj, _, _ = preprocess_adj(new_adj) if not util._SE_ else preprocess_adj_SE(new_adj)
                mutation_adj.append(new_adj)
                mutation_labels.append(torch.tensor([label]))  # the population's labels only take the form of tensors
                add_to_population(population,mutation_adj,mutation_labels,device) # add to population
                # then save population for each model evaluated
                torch.save(population,"Population_initialized") if not util._SE_ else torch.save(population,"Population_initialized_SE")	

        number_of_adjs_need_to_add_left -= num_mutate_adjs*mutate_num  # the remaining number of adjs that need to be added
    else:
        assert mutate_num==0, "There is no adjacency matrix to mutate!, please set mutate_num = 0"
    # now for the randomly generating part    
    for _ in range(number_of_adjs_need_to_add_left):
        random_adj = []
        labels_for_random_adj = []
        new_adj = randomly_generate(pruning_prob,num_node=num_node) 
        try:   
            label = train_nn(data,new_adj,device,lr=lr,epochs=epochs,batchsize=batchsize,num_cells=num_cells,ic=ic,
                            channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,stride=stride,
                            num_node=num_node,pool=pool,bottleneck_reduction=bottleneck_reduction,conv_ker=conv_ker,
                            avg_max_ker=avg_max_ker,skip=skip,num_classifier_nodes=num_classifier_nodes,
                            loss=loss,**pool_kwargs)
            if wandb is not None:
                wandb.log({"init_label":label})
        except:
                print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                continue
        # I comment the following line because population contains unpreprocessed adjs
        new_adj, _, _ = preprocess_adj(new_adj) if not util._SE_ else preprocess_adj_SE(new_adj)
        random_adj.append(new_adj)
        labels_for_random_adj.append(torch.tensor([label]))
        add_to_population(population,random_adj,labels_for_random_adj,device) # add to population
        # then save population for each model evaluated
        torch.save(population,"Population_initialized") if not util._SE_ else torch.save(population,"Population_initialized_SE")	
    return population

def initialize_population_save_each_model(initialize_num,data,device,
                        list_adj_to_mutate=None,mutate_num=0,pruning_prob=0.2,
                        lr=1e-2,epochs=20,batchsize=128,max_count=3,
                        num_cells=6,channel_at_cell=128,out_dim=5,dropout=0.0,
                        scheduler_type=torch.optim.lr_scheduler.OneCycleLR,T_max=16,
                        optimizer_type=torch.optim.AdamW,ic=12,
                        criterion_type=nn.functional.binary_cross_entropy,
                        loss=nn.functional.binary_cross_entropy,
                        weight_decay=1e-2,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],skip=False,
                        num_classifier_nodes=128*2,bottleneck_reduction=4,pool=AdaptiveConcatPool1d,
                        num_node=13,wandb=None,grad_clip=None,stride=False,**pool_kwargs
                        ):
    number_of_adjs_need_to_add_left = initialize_num    # count how many adjs needs to initialize left
    # population = torch.load("Population_initialized_SE_test",map_location=device)   # load previous run
    population = {}   
    if list_adj_to_mutate != None:
        num_mutate_adjs = len(list_adj_to_mutate)
        assert mutate_num != 0, "Must provide the number of mutation!"
        assert initialize_num>=num_mutate_adjs*mutate_num, "Number of initialize adjs cannot be smaller than the number of mutations!"
    
        for i in range(num_mutate_adjs):   # loop through all list_adj_to_mutate
            for _ in range(mutate_num):  
                mutation_adj = []
                mutation_labels = []
                # this func create a clone of list_adj_to_mutate[i] so it would not change the original 
                new_adj = mutate(list_adj_to_mutate[i])
                # try:
                label = train_nn(data,new_adj,device,lr=lr,epochs=epochs,batchsize=batchsize,num_cells=num_cells,ic=ic,
                            channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,stride=stride,num_node=num_node,
                            pool=pool,bottleneck_reduction=bottleneck_reduction,conv_ker=conv_ker,avg_max_ker=avg_max_ker,
                            skip=skip,num_classifier_nodes=num_classifier_nodes,optimizer_type=optimizer_type,
                            scheduler_type=scheduler_type,T_max=T_max,max_count=max_count,grad_clip=grad_clip,
                            loss=loss,weight_decay=weight_decay,criterion_type=criterion_type,**pool_kwargs)
                if wandb is not None:
                    wandb.log({"init_label":label})
                # except:
                #     print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                #     continue
                new_adj, _, _ = preprocess_adj(new_adj) if not util._SE_ else preprocess_adj_SE(new_adj)
                mutation_adj.append(new_adj)
                mutation_labels.append(torch.tensor([label]))  # the population's labels only take the form of tensors
                add_to_population(population,mutation_adj,mutation_labels,device) # add to population
                # then save population for each model evaluated
                torch.save(population,"Population_initialized") if not util._SE_ else torch.save(population,"Population_initialized_SE_test")	

        number_of_adjs_need_to_add_left -= num_mutate_adjs*mutate_num  # the remaining number of adjs that need to be added
    else:
        assert mutate_num==0, "There is no adjacency matrix to mutate!, please set mutate_num = 0"
    # now for the randomly generating part    
    for _ in range(number_of_adjs_need_to_add_left):
        random_adj = []
        labels_for_random_adj = []
        new_adj = randomly_generate(pruning_prob,num_node=num_node) 
        try:   
            label = train_nn(data,new_adj,device,lr=lr,epochs=epochs,batchsize=batchsize,num_cells=num_cells,ic=ic,
                            channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,stride=stride,num_node=num_node,
                            pool=pool,bottleneck_reduction=bottleneck_reduction,conv_ker=conv_ker,avg_max_ker=avg_max_ker,
                            skip=skip,num_classifier_nodes=num_classifier_nodes,optimizer_type=optimizer_type,
                            scheduler_type=scheduler_type,T_max=T_max,max_count=max_count,grad_clip=grad_clip,
                            loss=loss,weight_decay=weight_decay,criterion_type=criterion_type,**pool_kwargs)
            if wandb is not None:
                wandb.log({"init_label":label})
        except:
                print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
                continue
        # I comment the following line because population contains unpreprocessed adjs
        new_adj, _, _ = preprocess_adj(new_adj) if not util._SE_ else preprocess_adj_SE(new_adj)
        random_adj.append(new_adj)
        labels_for_random_adj.append(torch.tensor([label]))
        add_to_population(population,random_adj,labels_for_random_adj,device) # add to population
        # then save population for each model evaluated
        torch.save(population,"Population_initialized") if not util._SE_ else torch.save(population,"Population_initialized_SE_test")	
    return population

# get num_samples adjs from the dictionary randomly, the return will be a list of adj
def get_samples_in_population(population,num_samples):
    num_adj = len(population)
    assert num_adj >= num_samples, "num_samples is bigger than the size of the population!"
    sample_list = []
    permuted_index = np.random.permutation(num_adj)    # random permutation in range(num_adj)
    # for the first num_samples elements in the permutation: find key and value in the population
    for i in range(num_samples):   
        for j, key in enumerate(list(population.keys())):    
            if j==permuted_index[i]:
                sample_list.append(population[key])
    return sample_list

# return the best performing adjs
def return_best_adjs(population,num_samples,descending=True):
    list_labels = list(population.keys())
    assert num_samples <= len(list_labels), "num_samples is bigger than the size of the population!"
    list_labels.sort(reverse=descending)  # sort in descending order
    best_adjs = []
    for i in range(num_samples):
        best_adjs.append(population[list_labels[i]])
    return best_adjs


# recive a list of adjs, perform grad_ascent with all the predictors, train and return the new adjs and labels
# list_adj is a preprocessed_adj
def perform_grad_ascent_then_train(predictors,autoencoder,data,list_adj,device,lr=1e-2,lr_ascent=5e-1,lr_decay=0.5,
                                   epsilon=5e-2,min_lr=1e-5,epochs=20,batchsize=128,max_count=3,num_cells=6,channel_at_cell=128,
                                   out_dim=5,dropout=0.0,scheduler_type=torch.optim.lr_scheduler.OneCycleLR,T_max=16,
                                   optimizer_type=torch.optim.AdamW,criterion_type=nn.functional.binary_cross_entropy,num_node=13):
    list_adj = torch.stack(list_adj,dim=0)  # shape(bs,4,13,13)
    list_adj = list_adj.to(device)
    bs = len(list_adj)
    # freeze the 
    autoencoder.to(device)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    _, mu, _ = autoencoder(list_adj)
    mu.requires_grad = True   # enable gradient_flow
    new_mu, _ = grad_ascent_group(predictors,mu,device,lr=lr_ascent,lr_decay=lr_decay,epsilon=epsilon,min_lr=min_lr) # shape(bs,13,latent_dim)

    list_new_adj = autoencoder.decoder(new_mu)    # shape (bs,4,13,13)
    list_new_adj = list(torch.split(list_new_adj,1,dim=0))   # a list of bs tensors that have the shape of (1,4,13,13)        
    list_labels = []
    for i in range(bs):
        list_new_adj[i] = list_new_adj[i].squeeze(dim=0)  # shape (4,13,13)
        try:
            label = train_nn(data,list_new_adj[i],device,lr=lr,epochs=epochs,batchsize=batchsize,max_count=max_count,
                num_cells=num_cells,channel_at_cell=channel_at_cell,out_dim=out_dim,dropout=dropout,scheduler_type=scheduler_type,
                T_max=T_max,optimizer_type=optimizer_type,criterion_type=criterion_type,num_node=num_node)
        except:
            print("THE MODEL IS NOT IN THE RIGHT FORMAT! PROGRAMME WILL SKIP.")
            continue
        # I comment the following line because population contains unpreprocessed adjs
        # list_new_adj[i], _, _ = preprocess_adj(list_new_adj[i])
        list_labels.append(torch.tensor([label]))    # population labels must be tensors
    return list_new_adj, list_labels



# autoencoder = Model_VAE()
# predictors = [Predictor(16,8) for _ in range(3)]
# # population = {}
# list_adj = []
# for i in range(1):
#     adj = randomly_generate(0.2)
#     adj, _, _ = preprocess_adj(adj)
# #     # population[torch.tensor([i])] = adj
#     list_adj.append(adj)
# device = "cpu"
# self_train_predictors(predictors,autoencoder,list_adj,device,10,epochs=5,batch_size=2)
# train_predictors(predictors,autoencoder,population,device,epochs=5,batch_size=1)

# adj = randomly_generate(0.2)
# adj, _, _ = preprocess_adj(adj)
# data = ((torch.rand((40,12,1000)),torch.tensor(np.random.choice(2,(40,5),p=[0.2,0.8]))),(torch.rand((40,12,1000)),torch.tensor(np.random.choice(2,(40,5),p=[0.2,0.8]))))
# train_nn(data,adj,device,batchsize=1,epochs=6)


# population = initialize_population(2,data,device=device,list_adj_to_mutate=list_adj,mutate_num=1,batchsize=40,epochs=1)
# print(population)

# list_new_adj, list_labels = perform_grad_ascent_then_train(predictors,autoencoder,data,list_adj,device,batchsize=10,epochs=1)