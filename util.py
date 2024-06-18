import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from typing import Optional
import torchmetrics 
from sklearn.metrics import f1_score, roc_auc_score

_SE_ = False
_nodeSE_ = False
class SE(nn.Module):
    def __init__(self, out_dim, hidden_dim, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, int(hidden_dim * expansion), bias=False),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * expansion), out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

def cd_adaptiveconcatpool(relevant, irrelevant, module):
    mpr, mpi = module.mp.attrib(relevant,irrelevant)
    apr, api = module.ap.attrib(relevant,irrelevant)
    return torch.cat([mpr, apr], 1), torch.cat([mpi, api], 1)
def attrib_adaptiveconcatpool(self,relevant,irrelevant):
    return cd_adaptiveconcatpool(relevant,irrelevant,self)
class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`."
    def __init__(self, sz:Optional[int]=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)
    def attrib(self,relevant,irrelevant):
        return attrib_adaptiveconcatpool(self,relevant,irrelevant)

def maxpool(kernel,stride):
    return nn.MaxPool1d(kernel,stride=stride,padding=(kernel-1)//2)
def conv(in_planes, out_planes, kernel_size=3, stride=1, groups=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                    padding=(kernel_size-1)//2,bias = False,groups=groups)
def avgpool(kernel,stride):
    return nn.AvgPool1d(kernel,stride=stride,padding=(kernel-1)//2)

def add_op(i,kernel,in_channel = None,out_channel = None,stride = 1):
    '''
    have to check conv, in and out channel
    '''
    if i==0:
        return conv(in_channel,out_channel,kernel,stride)
    elif i==1:
        if in_channel == out_channel:
            return maxpool(kernel,stride)
        else:
            return nn.Sequential(*[maxpool(kernel,stride),conv(in_channel,out_channel,1)])
    elif i==2:
        if in_channel == out_channel:
            return avgpool(kernel,stride)
        else:
            return nn.Sequential(*[avgpool(kernel,stride),conv(in_channel,out_channel,1)])
    else:
        if in_channel==out_channel:
            if stride==1:
                return nn.Identity()
            else:
                return maxpool(stride,stride)
        else:
            return conv(in_channel,out_channel,stride,stride)
        
def add_op_SE(i,kernel,in_channel = None,out_channel = None,stride = 1,expansion=0.25):
    '''
    have to check conv, in and out channel
    '''
    if i==0:
        return conv(in_channel,out_channel,kernel,stride)
    elif i==1:
        if in_channel == out_channel:
            return maxpool(kernel,stride)
        else:
            return nn.Sequential(*[maxpool(kernel,stride),conv(in_channel,out_channel,1)])
    elif i==2:
        if in_channel == out_channel:
            return SE(in_channel,in_channel,expansion=expansion)
        else:
            return nn.Sequential(*[SE(in_channel,in_channel,expansion=expansion),conv(in_channel,out_channel,1)])
    else:
        if in_channel==out_channel:
            return nn.Identity()
        else:
            return conv(in_channel,out_channel,stride,stride)
        
def add_op1(i,kernel,in_channel = None,out_channel = None,stride = 1,expansion=0.25):
    '''
    have to check conv, in and out channel
    '''
    if i==0:
        return nn.Sequential(*[conv(in_channel,out_channel,kernel,stride),SE(out_channel,out_channel,expansion=expansion)])
    elif i==1:
        if in_channel == out_channel:
            return nn.Sequential(*[maxpool(kernel,stride),SE(in_channel,in_channel,expansion=expansion)])
        else:
            return nn.Sequential(*[maxpool(kernel,stride),conv(in_channel,out_channel,1),SE(out_channel,out_channel,expansion=expansion)])
    elif i==2:
        if in_channel == out_channel:
            return nn.Sequential(*[avgpool(kernel,stride),SE(in_channel,in_channel,expansion=expansion)])
        else:
            return nn.Sequential(*[avgpool(kernel,stride),conv(in_channel,out_channel,1),SE(out_channel,out_channel,expansion=expansion)])
    else:
        if in_channel==out_channel:
            return nn.Sequential(nn.Identity(),SE(in_channel,in_channel,expansion=expansion))
        else:
            return nn.Sequential(*[conv(in_channel,out_channel,stride,stride),SE(out_channel,out_channel,expansion=expansion)])
# find all the nodes that have input, and nodes that dont
# use breadth first search
def find_path(queue, adj, num_node):
    finish = []
    while len(queue):    
        node = queue.pop()
        if node not in finish:
            finish.append(node)
            for j in range(node+1,num_node-1): # the last node will always connect
                for i in range(4):
                    if adj[i,node,j] != 0:
                        queue.insert(0,j)
    nodes_that_dont_have_input = []
    for i in range(num_node-1):
        if i not in finish:
            nodes_that_dont_have_input.append(i)
    return nodes_that_dont_have_input, finish

        
def preprocess_adj(adj_,num_node=13,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9]):
    '''
    layer 0: node 0
    layer 0.5: node 1
    layer 1: node 2->6
    layer 2: node 7->11
    last: node 12

    the i is operation, j is the former node, k is the latter node

    NOTE: this function will not modify the adj_ in outer scope, but return the modified adj

    the threshold is a obtained by scaling kernel by loge function and then find the mean point between each pair of point,
    if the number in UNPREPROCESSED matrix (that were scaled, see line:...) lies within the range setted up by thresholds,
    it would take the value respectively as the kernel_size
    '''
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(avg_max_ker)
    conv_threshold = []    # if pass a certain threshold then set the adj[i,j,k] = kernel respectively
    avg_max_threshold = []
    for i in range(num_conv_ker):  
        assert conv_ker[i]%2 ==1, "convolution kernel size must be an odd number!"
        if i != num_conv_ker-1:
            conv_threshold.append((np.log(conv_ker[i])+np.log(conv_ker[i+1]))/2)   # find the middle point
        else:
            conv_threshold.append(np.ceil(np.log(conv_ker[i])))  # the last thresholds is the ceil(log(kernel))
    for i in range(num_avgmax_ker):
        assert avg_max_ker[i]%2 ==1, "averagepool or maxpool kernel size must be an odd number!"
        if i != num_avgmax_ker-1:
            avg_max_threshold.append((np.log(avg_max_ker[i])+np.log(avg_max_ker[i+1]))/2)   # find the middle point
        else:
            # NOTE: the term "-0.5" is because I want the scale from range 0->2.5 so that it wouldnt favor kernel_size=9 more than other kernelsizes
            avg_max_threshold.append(np.ceil(np.log(avg_max_ker[i]))-0.5) 

    assert (num_node-1-2)%2==0, "Layer 1 and 2 must have the same number of nodes!"
    num_node_in_layer_1_2 = (num_node-1-2)//2   # number of nodes in each layers
    adj = adj_.clone()  

    for j in range(num_node):     # upper triangular matrix
        for k in range(j+1):
            adj[:,j,k] = 0
    for k in range(num_node):
        for j in range(k):
            i = 0    
            local_max = adj[i,j,k]
            for i_ in range(1,4):    # extract the best operation
                if adj[i_,j,k] > local_max:
                    local_max = adj[i_,j,k]
                    adj[i,j,k] = 0
                    i = i_
                else:
                    adj[i_,j,k] = 0  
            if adj[i,j,k] > 0.5:    # if pass a threshold
                # 2->6
                if (j<=2+num_node_in_layer_1_2-1 and j>=2 and k<=2+num_node_in_layer_1_2-1 and k>=2):  
                    adj[i,j,k] = 0
                # 7->11
                elif (j<=2+2*num_node_in_layer_1_2-1 and j>=2+num_node_in_layer_1_2 and k<=2+2*num_node_in_layer_1_2-1 and k>=2+num_node_in_layer_1_2):   
                    adj[i,j,k] = 0
                # layer 2 cannot connect to the last node
                elif (j<=2+2*num_node_in_layer_1_2-1 and j>=2+num_node_in_layer_1_2 and k==num_node-1):  
                    adj[i,j,k] = 0
                else:
                    # this is when node 1 always connects with node 0 by a conv1 and connects with other by SE
                    if _nodeSE_ and j==0 and k==1 and i==0:
                        adj[:,0,1] = 0
                        adj[0,0,1] = 1  # must be conv1
                        continue
                    if i==3:    # noop or depthwise conv
                        adj[i,j,k] = 1
                    elif i==0:  # conv
                        temp = (adj[i,j,k]-0.5)/0.5*conv_threshold[-1] # -> project from range(0.5,1) to range(0,4)
                        temp_index = 0
                        while(temp>conv_threshold[temp_index]):   # find suitable index
                            temp_index+=1
                        adj[i,j,k] = conv_ker[temp_index]
                    else:       # avgpool and maxpool
                        temp = (adj[i,j,k]-0.5)/0.5*avg_max_threshold[-1] # -> project from range(0.5,1) to range(0,2.5)
                        temp_index = 0
                        while(temp>avg_max_threshold[temp_index]):   # find suitable index
                            temp_index+=1
                        adj[i,j,k] = avg_max_ker[temp_index]
            else:
                adj[i,j,k] = 0    # if not pass the threshold
    nodes_that_dont_have_input, nodes_that_HAVE_input = find_path([0],adj,num_node)
    for node in nodes_that_dont_have_input:
        for i in range(4):     # remove all the unnecessary connection
            adj[i,node,:] = 0
            adj[i,:,node] = 0
    adj = adj + torch.transpose(adj,1,2)  # to symmetric matrix

    # for node at the end of each path, if it doesnt connect with the last node then append to a list
    end_path = []
    for j in range(num_node-1):  # no need to check the last node
        if j in nodes_that_HAVE_input:
            check = False
            for k in range(j+1,num_node):
                for i in range(4):
                    if adj[i,j,k] != 0:
                        check = True
                        break
                if check:    # out loop if it is not the end node or it is not connected to the last node
                    break
            if not check:
                end_path.append(j) 
    nodes_that_HAVE_input.append(num_node-1)   # append the last node
    return adj, end_path, nodes_that_HAVE_input  # the previous list doesnt have the last node in it

def preprocess_adj_SE(adj_,num_node=13,conv_ker=[1,3,5,9,19,39],max_ker=[3,5,9]):
    '''
    layer 0: node 0
    layer 0.5: node 1
    layer 1: node 2->6
    layer 2: node 7->11
    last: node 12

    the i is operation, j is the former node, k is the latter node

    NOTE: this function will not modify the adj_ in outer scope, but return the modified adj

    the threshold is a obtained by scaling kernel by loge function and then find the mean point between each pair of point,
    if the number in UNPREPROCESSED matrix (that were scaled, see line:...) lies within the range setted up by thresholds,
    it would take the value respectively as the kernel_size
    '''
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(max_ker)
    conv_threshold = []    # if pass a certain threshold then set the adj[i,j,k] = kernel respectively
    max_threshold = []
    for i in range(num_conv_ker):  
        assert conv_ker[i]%2 ==1, "convolution kernel size must be an odd number!"
        if i != num_conv_ker-1:
            conv_threshold.append((np.log(conv_ker[i])+np.log(conv_ker[i+1]))/2)   # find the middle point
        else:
            conv_threshold.append(np.ceil(np.log(conv_ker[i])))  # the last thresholds is the ceil(log(kernel))
    for i in range(num_avgmax_ker):
        assert max_ker[i]%2 ==1, "averagepool or maxpool kernel size must be an odd number!"
        if i != num_avgmax_ker-1:
            max_threshold.append((np.log(max_ker[i])+np.log(max_ker[i+1]))/2)   # find the middle point
        else:
            # NOTE: the term "-0.5" is because I want the scale from range 0->2.5 so that it wouldnt favor kernel_size=9 more than other kernelsizes
            max_threshold.append(np.ceil(np.log(max_ker[i]))-0.5) 

    assert (num_node-1-2)%2==0, "Layer 1 and 2 must have the same number of nodes!"
    num_node_in_layer_1_2 = (num_node-1-2)//2   # number of nodes in each layers
    adj = adj_.clone()  

    for j in range(num_node):     # upper triangular matrix
        for k in range(j+1):
            adj[:,j,k] = 0
    for k in range(num_node):
        for j in range(k):
            i = 0    
            local_max = adj[i,j,k]
            for i_ in range(1,4):    # extract the best operation
                if adj[i_,j,k] > local_max:
                    local_max = adj[i_,j,k]
                    adj[i,j,k] = 0
                    i = i_
                else:
                    adj[i_,j,k] = 0  
            if adj[i,j,k] > 0.5:    # if pass a threshold
                # 2->6
                if (j<=2+num_node_in_layer_1_2-1 and j>=2 and k<=2+num_node_in_layer_1_2-1 and k>=2):  
                    adj[i,j,k] = 0
                # 7->11
                elif (j<=2+2*num_node_in_layer_1_2-1 and j>=2+num_node_in_layer_1_2 and k<=2+2*num_node_in_layer_1_2-1 and k>=2+num_node_in_layer_1_2):   
                    adj[i,j,k] = 0
                # layer 2 cannot connect to the last node
                elif (j<=2+2*num_node_in_layer_1_2-1 and j>=2+num_node_in_layer_1_2 and k==num_node-1):  
                    adj[i,j,k] = 0
                else:
                    if i==3 or i==2:    # noop or depthwise conv
                        adj[i,j,k] = 1
                    elif i==0:  # conv
                        temp = (adj[i,j,k]-0.5)/0.5*conv_threshold[-1] # -> project from range(0.5,1) to range(0,4)
                        temp_index = 0
                        while(temp>conv_threshold[temp_index]):   # find suitable index
                            temp_index+=1
                        adj[i,j,k] = conv_ker[temp_index]
                    else:       # avgpool and maxpool
                        temp = (adj[i,j,k]-0.5)/0.5*max_threshold[-1] # -> project from range(0.5,1) to range(0,2.5)
                        temp_index = 0
                        while(temp>max_threshold[temp_index]):   # find suitable index
                            temp_index+=1
                        adj[i,j,k] = max_ker[temp_index]
            else:
                adj[i,j,k] = 0    # if not pass the threshold
    nodes_that_dont_have_input, nodes_that_HAVE_input = find_path([0],adj,num_node)
    for node in nodes_that_dont_have_input:
        for i in range(4):     # remove all the unnecessary connection
            adj[i,node,:] = 0
            adj[i,:,node] = 0
    adj = adj + torch.transpose(adj,1,2)  # to symmetric matrix

    # for node at the end of each path, if it doesnt connect with the last node then append to a list
    end_path = []
    for j in range(num_node-1):  # no need to check the last node
        if j in nodes_that_HAVE_input:
            check = False
            for k in range(j+1,num_node):
                for i in range(4):
                    if adj[i,j,k] != 0:
                        check = True
                        break
                if check:    # out loop if it is not the end node or it is not connected to the last node
                    break
            if not check:
                end_path.append(j) 
    nodes_that_HAVE_input.append(num_node-1)   # append the last node
    return adj, end_path, nodes_that_HAVE_input  # the previous list doesnt have the last node in it
    
class Create_cell(pl.LightningModule):
    def __init__(self,adj,ic,channel_at_cell=128,channel_at_node=None,stride=1,num_node=13,bottleneck_reduction=4,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9]):
        super().__init__()
        assert (num_node-1-2)%2==0, "Layer 1 and 2 must have the same number of nodes!"
        self.adj, self.end_path, self.use_node = preprocess_adj(adj,num_node,conv_ker=conv_ker,avg_max_ker=avg_max_ker) if not _SE_ else preprocess_adj_SE(adj,num_node,conv_ker=conv_ker,max_ker=avg_max_ker)  # preprocess the adjacency matrix
        self.bn_relu = nn.Sequential(*[nn.BatchNorm1d(channel_at_cell),nn.ReLU()])  # add batchnorm and activation
        self.channel_at_cell = channel_at_cell
        self.bottleneck_reduction = bottleneck_reduction
        self.stride = stride
        self.upchannel = True if stride ==2 else False
        num_node_in_layer_1_2 = (num_node-1-2)//2
        nodes_are_bottleneck = [1,2,2+num_node_in_layer_1_2]  # 1,2,7, these are the bottleneck nodes
        self.last_node = str(num_node-1)   # a string of last node's index, this is for keys in the dictionaries
        self.num_node = num_node

        '''
        There are 13 nodes, node 0 is the input node with outchannel is 128. The node 1 is special node with outchannel 
        of 32 and all nodes that take the output of node 1 as input or have a directed path from 1 to that node will have 
        outchannel of 32 except for the last node (12, wil always have 128 channels).
        
        Node 7-11 will always connect to the last node, but apply no operation. Instead, if there are outputs of 7-11 
        that have 32 channels, they will be concatenated and added with the 128-channel-node, thus the output will
        be a 128-channel-tensor. If there are only 32-channel-node, they will be concatenated and feed through 
        depthwise convolutional layer to output a 128-channel-tensor.

        Other nodes beside from 7-11 that connects to the last node will be treat normally
        
        '''
        self.use_node.remove(0)       # IMPORTANT: this is for convenient, the first node still be used
        self.use_node.sort()          # INPORTANT: the smaller nodes need to be updated first

        self.list_op = nn.ModuleDict()
        for i in self.use_node:
            self.list_op[str(i)] = nn.ModuleDict() # append operation dictionary

        check_if_first = True    # check if this is the first cell, this means the channel_at_node != None
        self.channel_at_node = channel_at_node
        if self.channel_at_node==None:
            self.channel_at_node = {}      # output channel of each node
            for i in range(2,num_node):
                self.channel_at_node[i] = channel_at_cell  # initialize channel, will be updated later
            for j in nodes_are_bottleneck:
                self.channel_at_node[j] = channel_at_cell//bottleneck_reduction  # set the bottleneck nodes
            check_if_first = False
        self.channel_at_node[0] = ic    

        self.extract_adj(check_if_first) # update channel and add layer 

        self.countsmall = 0     # number of path that has channel_at_cell//4 channels
        self.countbig = 0       # number of path that has channel_at_cell channels
        for node in self.end_path: 
            if self.channel_at_node[node]==channel_at_cell//bottleneck_reduction:
                self.countsmall +=1
            elif self.channel_at_node[node]==channel_at_cell:
                self.countbig +=1
        if self.countsmall > bottleneck_reduction:   # this is: if there are too many nodes that have channel_at_cell//4 channels
            self.down_channel = conv(channel_at_cell//bottleneck_reduction*self.countsmall,
                                        channel_at_cell,kernel_size=1)   # the last node must have 128 channels

    def forward(self,x):
        output = {'0':x}  # the output of all the nodes, the output of the first node is x
        for j in self.use_node: # the last node is different!, also we removed node 0 bc of this following loop
                temp = []
                if len(self.list_op[str(j)]) != 0:    # this is for the last node, if there is no node that connect to the last node , the line 197 would raise error if this line if-else didnt exist
                    for k in self.list_op[str(j)].keys(): # check all previous nodes
                        temp.append(self.list_op[str(j)][k](output[k]))  # output of the previous nodes will be feedforwarded
                    output[str(j)] = torch.sum(torch.stack(temp,dim=0),dim=0) # stack all the temporal output and then sum
                else:
                    output[str(j)] = 0

        # the last node:
        output_32 = []
        temp = []
        for node in self.end_path:   # concat or add the second-to-last layer
                if self.channel_at_node[node]==self.channel_at_cell:
                    output[self.last_node] += output[str(node)]
                else:
                    output_32.append(output[str(node)])
        if (self.countsmall>0 and self.countsmall<=self.bottleneck_reduction):  # 0<x<4
            for _ in range(self.bottleneck_reduction-self.countsmall):
                output_32.append(torch.zeros_like(output_32[0]))   # add zero so that the list has 4 elements
            output[self.last_node] += torch.concatenate(output_32,dim=1)  # concat to 128 channel, shape (bs,128,1000)
        elif self.countsmall > self.bottleneck_reduction:
            output[self.last_node] += self.down_channel(torch.concatenate(output_32,dim=1))  # downchannel to 128

        # if count == 0, only need to add up all the elements in output_128
        return self.bn_relu(output[self.last_node])
    

    def extract_adj(self,check_if_first):
        # channel 128, bottleneck 32
        for j in self.use_node: 
            temp_unordered_dict = {}
            for k in range(j): # only check if this is forward connection, this will be skipped if j=0, k is former node, j is latter node
                for i in range(4): # 4 operations
                    if self.adj[i,j,k] != 0:
                        # update channel, all other node except for node 1 that will have 32 channels
                        # the first node can have 12 channels as for 12 leads
                        # this only run if channel_at_node is None -> check_if_first = False
                        if not check_if_first and j!=self.num_node-1:
                            if not self.upchannel:   
                                self.channel_at_node[j] = min(self.channel_at_node[k],self.channel_at_node[j])  
                            else:
                                if k==0:
                                    self.channel_at_node[j] = self.channel_at_cell*2
                                else:
                                    self.channel_at_node[j] = min(self.channel_at_node[k],self.channel_at_node[j]) 
            for k in range(j):
                for i in range(4):
                    if self.adj[i,j,k] != 0:

                        if _nodeSE_ and k==1:
                            temp_unordered_dict[str(k)] = add_op1(i,kernel=self.adj[i,j,k].int().item(),
                                                in_channel=self.channel_at_node[k],
                                                out_channel=self.channel_at_node[j],stride=self.stride if k==0 else 1) 
                            continue

                        temp_unordered_dict[str(k)] = add_op(i,kernel=self.adj[i,j,k].int().item(),
                                                in_channel=self.channel_at_node[k],
                                                out_channel=self.channel_at_node[j],stride=self.stride if k==0 else 1) if not _SE_ else add_op_SE(i,kernel=self.adj[i,j,k].int().item(),
                                                                                                        in_channel=self.channel_at_node[k],
                                                                                                        out_channel=self.channel_at_node[j],stride=self.stride if k==0 else 1)
            key =  list(temp_unordered_dict.keys())
            key.sort()       # Important: sorting all the keys in the right order as the forward method will compute the output of lower-index-node first
            # Important: the outer dict dont need to be sorted as I have sorted the self.use_node before
            self.list_op[str(j)] = nn.ModuleDict({k: temp_unordered_dict[k] for k in key}) 

class Create_model(pl.LightningModule):
    def __init__(self,adj,num_cells,ic=12,channel_at_cell=128,out_dim=5,dropout=0.0,lr=1e-2,stride = None,num_node=13,
                bottleneck_reduction=4,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],skip=False,num_classifier_nodes=128*2,
                loss=nn.functional.binary_cross_entropy,pool=AdaptiveConcatPool1d,**pool_kwargs):
        super().__init__()
        self.adj = adj
        self.num_cells = num_cells
        # self.accuracy = torchmetrics.Accuracy('multiclass',num_classes=out_dim)
        self.loss = loss
        assert num_cells>=2, "num_cell must be more than 2!"
        self.skip = skip
        if stride:
            self.stride = stride
        else:
            self.stride = [1 for _ in range(num_cells)]
        self.lr = lr
        self.store = []

        self.cells = nn.ModuleList([Create_cell(adj,channel_at_cell,channel_at_cell,stride=self.stride[1],channel_at_node=None,
                                                num_node=num_node,bottleneck_reduction=bottleneck_reduction,
                                                conv_ker=conv_ker,avg_max_ker=avg_max_ker,
                                                )])
        # take the channel_at_node then initialize everything else, save time and also crucial for constructing the first cell
        self.channel_at_node = self.cells[0].channel_at_node
        self.cells.insert(0,Create_cell(adj,ic,channel_at_cell,channel_at_node=self.channel_at_node,stride=self.stride[0],
                                        num_node=num_node,bottleneck_reduction=bottleneck_reduction,
                                        conv_ker=conv_ker,avg_max_ker=avg_max_ker,
                                        ))   
        for i in range(self.num_cells-2):
            self.cells.append(Create_cell(adj,channel_at_cell,channel_at_cell,
                                        channel_at_node=self.channel_at_node,stride=self.stride[2+i],
                                        num_node=num_node,bottleneck_reduction=bottleneck_reduction,
                                        conv_ker=conv_ker,avg_max_ker=avg_max_ker,
                                        ))
        if pool_kwargs:
            self.pool = pool(**pool_kwargs)
        else:
            self.pool = pool(1)  # adaptiveconcatpool1d(1)

        if self.skip:
            skipconnect = [conv(ic,channel_at_cell,1)]
            for _ in range(num_cells//3-1):
                skipconnect.append(conv(channel_at_cell,channel_at_cell,1))
            self.skipconnect = nn.ModuleList(skipconnect)

        classify = [nn.Flatten()]
        classify.append(nn.Linear(num_classifier_nodes,channel_at_cell))
        classify.append(nn.ReLU())
        classify.append(nn.Linear(channel_at_cell,out_dim))
        classify.append(nn.Dropout(dropout))  # add dropout
        # classify.append(nn.Sigmoid())
        # classify.append(nn.Softmax())
        self.classify = nn.Sequential(*classify)
    def forward(self,x):
        for i in range(self.num_cells):
            if self.skip and i<6 and i%3==0:
                res = x.clone()
                res = self.skipconnect[i//3](res)
            x = self.cells[i](x)
            if self.skip and i%3==2:
                x = res+x
        x = self.pool(x)
        return self.classify(x)
    def training_step(self,batch,batch_idx):
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        loss = self.loss(y_hat,y)
        self.log("train_loss",loss)
        
        return loss
    def validation_step(self,batch,batch_idx):
        self.eval()
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        
        y_pred = torch.argmax(y_hat, dim=1)
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='macro')  # Use 'micro' or 'weighted' as needed
        self.log("val_f1_score", f1)
        self.accuracy(y_hat, y)
        self.log("val_acc_step", self.accuracy)
        
        # self.log("val_auc_step", roc_auc_score(y.cpu(),y_hat.cpu()))
        
    def test_step(self,batch,batch_idx):
        self.eval()
        x, y = batch
        x = x.float()
        y = y.long()
        y_hat = self.forward(x)
        
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        
        # Calculate F1 score
        y_pred = torch.argmax(y_hat, dim=1)
        f1 = f1_score(y.cpu(), y_pred.cpu(), average='macro')  # Use 'micro' or 'weighted' as needed
        self.log("test_f1_score", f1)
        self.accuracy(y_hat, y)
        self.log("test_acc_step", self.accuracy)
        
        # self.log("test_auc_step", roc_auc_score(y.cpu(),y_hat.cpu()))
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # return optimizer
        stepping_batches = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=stepping_batches)
        return [optimizer], [scheduler]
    
# from kernel size to the sigmoid representation, eg.: 39 (conv) -> 3.5 -> 3.5/4
# i is operation_index, num is kernel_size
def reverse_to_sigmoid(index,num,conv_threshold,avg_max_threshold,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9]):
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(avg_max_ker)
    # reversing:
    if index==0:
        for j in range(num_conv_ker):
            if conv_ker[j]==num:
                # project from range(0,4) to range(0.5,1)
                return (conv_threshold[j]-0.01)/(2*conv_threshold[-1]) + 0.5
    elif index==3:
        return 1
    else:
        for j in range(num_avgmax_ker):
            if avg_max_ker[j]==num:
                # project from range(0,4) to range(0.5,1)
                return (avg_max_threshold[j]-0.01)/(2*avg_max_threshold[-1]) + 0.5
            
def reverse_to_sigmoid_SE(index,num,conv_threshold,avg_max_threshold,conv_ker=[1,3,5,9,19,39],max_ker=[3,5,9]):
    num_conv_ker = len(conv_ker)
    num_max_ker = len(max_ker)
    # reversing:
    if index==0:
        for j in range(num_conv_ker):
            if conv_ker[j]==num:
                # project from range(0,4) to range(0.5,1)
                return (conv_threshold[j]-0.01)/(2*conv_threshold[-1]) + 0.5
    elif index==3 or index==2:
        return 1
    else:
        for j in range(num_max_ker):
            if max_ker[j]==num:
                # project from range(0,4) to range(0.5,1)
                return (avg_max_threshold[j]-0.01)/(2*avg_max_threshold[-1]) + 0.5


# this function projects INITIALIZED AND PREPROCESSED model to UNPREPROCESSED ones
def reverse_to_sigmoid_random(adj,num_node=13,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9]):
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(avg_max_ker)
    conv_threshold = []    # if pass a certain threshold then set the adj[i,j,k] = kernel respectively
    avg_max_threshold = []
    conv_threshold.append(0)
    avg_max_threshold.append(0)
    for i in range(num_conv_ker):  
        assert conv_ker[i]%2 ==1, "convolution kernel size must be an odd number!"
        if i != num_conv_ker-1:
            conv_threshold.append((np.log(conv_ker[i])+np.log(conv_ker[i+1]))/2)   # find the middle point
        else:
            conv_threshold.append(np.ceil(np.log(conv_ker[i])))  # the last thresholds is the ceil(log(kernel))
    # NOTE: conv_threshold = [0,th1,th2,...,max] (len = num_conv_ker + 1)
    
    for i in range(num_avgmax_ker):
        assert avg_max_ker[i]%2 ==1, "averagepool or maxpool kernel size must be an odd number!"
        if i != num_avgmax_ker-1:
            avg_max_threshold.append((np.log(avg_max_ker[i])+np.log(avg_max_ker[i+1]))/2)   # find the middle point
        else:
            # NOTE: the term "-0.5" is because I want the scale from range 0->2.5 so that it wouldnt favor kernel_size=9 more than other kernelsizes
            avg_max_threshold.append(np.ceil(np.log(avg_max_ker[i]))-0.5) 
    # NOTE: avg_max_threshold = [0,th1,th2,...,max] (len = num_avgmax_ker + 1)

    adj1 = adj.clone()   # create a clone of adj
    for j in range(num_node):
        for k in range(j+1):
            adj1[:,j,k] = 0   # delete lower triagular matrix
    for i in range(4):    
        for j in range(num_node):   # upper matrix
            for k in range(j+1,num_node):
                if adj1[i,j,k] != 0:
                    if i==0:    # conv
                        for index,kernel_size in enumerate(conv_ker):
                            if kernel_size == adj1[i,j,k]:
                                # randomly project to range between 2 thresholds
                                adj1[i,j,k] = np.random.uniform(low=conv_threshold[index],high=conv_threshold[index+1])/8+0.5
                                break
                    elif i==3:  # identity
                        adj1[i,j,k] = np.random.uniform(low=0.5, high=1)
                    else:       # avg and max
                        for index,kernel_size in enumerate(avg_max_ker):
                            if kernel_size == adj1[i,j,k]:
                                # randomly project to range between 2 thresholds
                                adj1[i,j,k] = np.random.uniform(low=avg_max_threshold[index],high=avg_max_threshold[index+1])/5 +0.5
                                break
                else:
                    adj1[i,j,k] = np.random.uniform(low=0,high=0.5)
    return adj1 + torch.transpose(adj1,1,2)


# IMPORTANT: mutate the PREPROCESSED ADJACENCY MATRIX, not the randomly generated one
# NOTE: this function will not modify the adj in outer scope, but return the modified adj1
def mutate(adj,num_node=13,prob=0.15,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9]):
    mutate_matrix = np.random.choice(2,(num_node,num_node),p=[1-prob,prob])   # create a (13,13) matrix with elements be either 0 or 1
    adj1 = adj.clone()       # we dont want to change the adj 
    # reverse to sigmoid form
    adj1 = reverse_to_sigmoid_toan_bo(adj1,num_node=num_node,conv_ker=conv_ker,avg_max_ker=avg_max_ker)

    for j in range(num_node):     # remove the lower propotion of mutate_matrix and adj1
        for k in range(j+1):
            mutate_matrix[j,k] = 0  
            for i in range(4):
                adj1[i,j,k] = 0
    for k in range(num_node):      # scanning the upper propotion
        for j in range(k):
            # if this is the node that is prohibited by the preprocessing method then it would not make any change!
            if mutate_matrix[j,k]==1: 
                check = False
                for i in range(4):
                    if adj1[i,j,k] != 0:
                        check = True
                        break     
                # "i" will contain the index of the operation that adj1[:,j,k] contains (if there is any)
                if check:
                    # change_op will indicate whether we change operation or we change the kernel, if 1->change operation
                    change_op = np.random.choice(2)   # uniform distribution that return either 0 or 1
                    if change_op==1:
                        temp_op = []
                        for t in range(4):   # if change op -> append all ops other than previous op
                            if (t!=i):
                                temp_op.append(t)
                        op_index = np.random.choice(temp_op)  # uniform distribution that return op in a temp_op
                        adj1[op_index,j,k] = np.random.uniform(low=0.5,high=1)   # this would create a operation in the preprocessing
                    else:
                        adj1[i,j,k] = np.random.uniform(low=0.5,high=1)  # this could result in the same kernel as before!
                else:
                    temp_op = []
                    for t in range(4):
                        temp_op.append(t)
                    op_index = np.random.choice(temp_op)   # uniform distribution that return op in a temp_op
                    adj1[op_index,j,k] = np.random.uniform(low=0.5,high=1)    # this would create a operation in the preprocessing
    
    # yes we need to call preprocess_adj(adj1), but we will once we call CreateCell() so there is no need to do so right now
    adj1 = adj1 + torch.transpose(adj1,1,2)  # symmetric matrix
    return adj1


# randomly generate adjacency matrix
# NOTE: the returned matrix has not been preprocessed
def randomly_generate(pruning_prob = None,num_node=13):
    if pruning_prob == None:
        adj = torch.rand((4,num_node,num_node))
        return adj
    else:
        adj = torch.rand((4,num_node,num_node))
        pruning_matrix = np.random.choice(2,(num_node,num_node),[1-pruning_prob,pruning_prob])
        for k in range(num_node):     # the upper portion of the matrix because the preprocessing method only scan upper matrix
            for j in range(k):
                if pruning_matrix[j,k] == 1:  # pruning adj
                    adj[:,j,k] = 0
        for j in range(num_node):    # delete lower triagular matrix
            for k in range(j+1):
                adj[:,j,k] = 0
        adj = adj + torch.transpose(adj,1,2)  # symmetric matrix
        return adj

# NOTE: list_adj must contain adjs that have already been preprocessed  
def generate_for_searching_phase(list_adj,num_outcome,mutate_prob=0.8,crossover_prob=0.0,dom_prob=0.7,
                                sub_prob=0.3,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],pruning_prob=0.2,
                                cross_mutate_prob=0.08,num_node=13):
    # create an array of shape (num_outcome) of 0 or 1 or 2, if 1 then mutate, 0 then generate, 2 then crossover
    prob = np.random.choice(3,num_outcome,p=[1-mutate_prob-crossover_prob,mutate_prob,crossover_prob]) 
    num_mat = len(list_adj)
    list_new_adj = []
    for i in range(num_outcome):
        if prob[i]==1:
            adj_index = np.random.choice(num_mat,1)     # draw a random index from [0,1,...,num_mat-1]
            adj = mutate(list_adj[adj_index.item()],num_node=num_node,conv_ker=conv_ker,avg_max_ker=avg_max_ker) 
            list_new_adj.append(adj.clone())  # autoencoder recieve unpreprocessed adj
        elif prob[i]==2:
            adj_index1 = np.random.choice(num_mat,1)     # draw a random index from [0,1,...,num_mat-1]
            adj_index2 = np.random.choice(num_mat,1)
            while adj_index2==adj_index1:   # index2 != index1
                adj_index2 = np.random.choice(num_mat,1)
            if np.random.choice(2,1):
                adj = cross_over2(list_adj[adj_index1.item()],list_adj[adj_index2.item()],dom_prob=dom_prob,
                                cross_mutate_prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker,num_node=num_node)  
            else:
                adj = cross_over(list_adj[adj_index1.item()],list_adj[adj_index2.item()],dom_prob=dom_prob,sub_prob=sub_prob,
                            cross_mutate_prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker,num_node=num_node)  
            list_new_adj.append(adj.clone())  # autoencoder recieve unpreprocessed adj
        else:
            adj = randomly_generate(pruning_prob,num_node=num_node)   # preprocessing adj because randomly generate didnt
            list_new_adj.append(adj.clone())
    return list_new_adj

# NOTE: list_adj must contain adjs that have already been preprocessed  
def generate_for_finetune_phase(list_adj,num_outcome,mutate_prob=0.5,crossover_prob=0.5,dom_prob=0.7,
                                sub_prob=0.3,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],pruning_prob=0.2,
                                cross_mutate_prob=0.08,num_node=13):
    # create an array of shape (num_outcome) of 0 or 1 or 2, if 1 then mutate, 0 then generate, 2 then crossover
    prob = np.random.choice(3,num_outcome,p=[1-mutate_prob-crossover_prob,mutate_prob,crossover_prob]) 
    num_mat = len(list_adj)
    list_new_adj = []
    for i in range(num_outcome):
        if prob[i]==1:
            adj_index = np.random.choice(num_mat,1)     # draw a random index from [0,1,...,num_mat-1]
            adj = mutate(list_adj[adj_index.item()],num_node=num_node)  
            list_new_adj.append(adj.clone())  # autoencoder recieve unpreprocessed adj
        elif prob[i]==2:
            adj_index1 = np.random.choice(num_mat,1)     # draw a random index from [0,1,...,num_mat-1]
            adj_index2 = np.random.choice(num_mat,1)
            while adj_index2==adj_index1:   # index2 != index1
                adj_index2 = np.random.choice(num_mat,1)
            if np.random.choice(2,1):
                adj = cross_over2(list_adj[adj_index1.item()],list_adj[adj_index2.item()],dom_prob=dom_prob,
                                cross_mutate_prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker,num_node=num_node)  
            else:
                adj = cross_over(list_adj[adj_index1.item()],list_adj[adj_index2.item()],dom_prob=dom_prob,sub_prob=sub_prob,
                            cross_mutate_prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker,num_node=num_node) 
            list_new_adj.append(adj.clone())  # autoencoder recieve unpreprocessed adj
        else:
            adj = randomly_generate(pruning_prob,num_node=num_node)   # preprocessing adj because randomly generate didnt
            list_new_adj.append(adj.clone())
    return list_new_adj

# generate a finetuning mutation of best_adj
def generate_finetuning_mutation(best_adj,num_outcome,prob=0.3,max_conv_ker=45, max_avgmax_ker=9,num_node=13):
    num_mat = len(best_adj)
    list_new_adj = []
    for i in range(num_outcome):
        adj_index = np.random.choice(num_mat,1)     # draw a random index from [0,1,...,num_mat-1]
        adj = finetune_NAS(best_adj[adj_index.item()],num_node=num_node,prob=prob,max_conv_ker=max_conv_ker,max_avgmax_ker=max_avgmax_ker)  
        list_new_adj.append(adj.clone())  # autoencoder recieve unpreprocessed adj
    return list_new_adj

# this function receive preprocessed matrix and it will randomly changing kernelsize
def finetune_NAS(best_adj,num_node=13,prob=0.3,max_conv_ker=45, max_avgmax_ker=9):
    # create a dictionary of mutation kernel for every existing kernel in conv_ker and avg_max_ker
    conv_ker = []    # we will find all the existing kernelsize in best adj then append to this list
    avg_max_ker = []
    for j in range(num_node):
        for k in range(j):     # the lower matrix
            for i in range(3):
                if best_adj[i,j,k]!=0:
                    if i==0 and best_adj[i,j,k].item() not in conv_ker:
                        conv_ker.append(best_adj[i,j,k].item())
                    elif (i==1 or i==2) and  best_adj[i,j,k].item() not in avg_max_ker:
                        avg_max_ker.append(best_adj[i,j,k].item())
    if max_conv_ker not in conv_ker:   # append the max kernel size
        conv_ker.append(max_conv_ker)
    if 1 not in conv_ker:    # append the min kernel size
        conv_ker.append(1)
    if max_avgmax_ker not in avg_max_ker:
        avg_max_ker.append(max_avgmax_ker)
    if 3 not in avg_max_ker:
        avg_max_ker.append(3)
    conv_ker.sort()  # sort in ascending order
    avg_max_ker.sort()

    # append all the necessary kernelsize for finetuning
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(avg_max_ker)
    all_mutate_conv_ker = {}
    all_mutate_avgmax_ker = {}
    for ker in conv_ker:
        all_mutate_conv_ker[ker] = []
    for i, ker in enumerate(conv_ker):
        if i < num_conv_ker-1:  # not the last element
            next_ker = conv_ker[i+1]
            if ker + 2 >= next_ker:  # if there is no gap between kernelsize in conv_ker ie, 3->5
                all_mutate_conv_ker[ker].append(next_ker)
                all_mutate_conv_ker[next_ker].append(ker)
            else:
                threshold = (ker+next_ker)//2   # the middle value
                if threshold%2==0:
                    threshold += 1    # ensure threshold must be an odd number
                temp_ker = ker+2
                all_mutate_conv_ker[ker].append(threshold)  # append the threshold bc the while loop will not append it to the list
                while temp_ker<next_ker:
                    if temp_ker <threshold:
                        all_mutate_conv_ker[ker].append(temp_ker)
                    else:
                        all_mutate_conv_ker[next_ker].append(temp_ker)
                    temp_ker += 2 

    for ker in avg_max_ker:
        all_mutate_avgmax_ker[ker] = []
    for i, ker in enumerate(avg_max_ker):
        if i < num_avgmax_ker-1:  # not the last element
            next_ker = avg_max_ker[i+1]
            if ker + 2 >= next_ker:  # if there is no gap between kernelsize in conv_ker ie, 3->5
                all_mutate_avgmax_ker[ker].append(next_ker)
                all_mutate_avgmax_ker[next_ker].append(ker)
            else:
                threshold = (ker+next_ker)//2   # the middle value
                if threshold%2==0:
                    threshold += 1    # ensure threshold must be an odd number
                temp_ker = ker+2
                all_mutate_avgmax_ker[ker].append(threshold)  # append the threshold bc the while loop will not append it to the list
                while temp_ker<next_ker:
                    if temp_ker <threshold:
                        all_mutate_avgmax_ker[ker].append(temp_ker)
                    else:
                        all_mutate_avgmax_ker[next_ker].append(temp_ker)
                    temp_ker += 2 

    best_adj1 = best_adj.clone()   # so that it wont change the best_adj
    for j in range(num_node):   # clear out the lower matrix
        for k in range(j):
            best_adj1[:,j,k] = 0
    for i in range(3):   # conv, max ,avg
        for j in range(num_node):    #  process the upper matrix
            for k in range(j+1,num_node):
                if best_adj1[i,j,k]!=0:
                    mutate_or_not = np.random.choice(2,1,p=[1-prob,prob])
                    if mutate_or_not == 0:
                        continue
                    if i==0:
                        for ker in all_mutate_conv_ker.keys():
                            if best_adj1[i,j,k] == ker:
                                num_option = len(all_mutate_conv_ker[ker])
                                index = np.random.choice(num_option,1)  # draw random index from range(0,num_option-1)
                                best_adj1[i,j,k] = all_mutate_conv_ker[ker][index.item()]
                                break
                    else:
                        for ker in all_mutate_avgmax_ker.keys():
                            if best_adj1[i,j,k] == ker:
                                num_option = len(all_mutate_avgmax_ker[ker])
                                index = np.random.choice(num_option,1)  # draw random index from range(0,num_option-1)
                                best_adj1[i,j,k] = all_mutate_avgmax_ker[ker][index.item()]
                                break

    best_adj1 = reverse_to_sigmoid_toan_bo(best_adj1,num_node=num_node,conv_ker=[i for i in range(1,max_conv_ker+1,2)],
                                        avg_max_ker=[i for i in range(3,max_avgmax_ker+1,2)])
    
    best_adj1 = best_adj1 + torch.transpose(best_adj1,1,2)
    return best_adj1

def scale(label,max=0.940,min=0.920):
    range_ = max - min
    return (label-min)/range_

# this function find all the path that connects id to the first node
def append_path(id, adj, array, store_path):
    if id == 0:
        store_path.append(array.copy())
    for node in range(id):  # traverse through all the previous nodes
        for i in range(4):
            if adj[i,node,id] != 0:  # if connect to the current node
                array.append(node)
                append_path(node,adj,array,store_path)  # recursive call 
                array.pop()  # remove the node that has been appended

# this function extract all the path in the end_path list
def extract_path(adj,num_node=13):
    _, nodes_that_HAVE_input = find_path([0],adj,num_node)
    # for node at the end of each path, if it doesnt connect with the last node then append to a list
    end_path = []
    for j in range(num_node-1):  # no need to check the last node
        if j in nodes_that_HAVE_input:
            check = False
            for k in range(j+1,num_node):
                for i in range(4):
                    if adj[i,j,k] != 0:
                        check = True
                        break
                if check:    # out loop if it is not the end node or it is not connected to the last node
                    break
            if not check:
                end_path.append(j) 
    end_path.append(num_node-1)  # the last node is also the end path

    # store_path stores all the path that ends with end_path's nodes, store_path_dic converts store_path to a dictionary
    store_path = []
    store_path_dic = {}
    for node in end_path:
        append_path(node,adj,[node],store_path)
        store_path_dic[str(node)] = []
    for path in store_path:
        end_node = path[0]
        path.remove(end_node) # remove the end_node
        store_path_dic[str(end_node)].append(path)
    return store_path_dic

def cross_over(adj0,adj1,dom_prob=0.8,sub_prob=0.3,cross_mutate_prob=0.08,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],num_node=13):
    index = np.random.choice(2,1)   # choose an architecture to be the domination
    list_adj = []
    if index==0:
        list_adj.append(adj0)
        list_adj.append(adj1)
    else:
        list_adj.append(adj1)
        list_adj.append(adj0) 
    path_dom = extract_path(list_adj[0],num_node=num_node)
    path_sub = extract_path(list_adj[1],num_node=num_node)
    adj = torch.zeros_like(adj0)
    # add the dominate path first
    for end_node in path_dom.keys():
        for path in path_dom[end_node]:
            if np.random.choice(2,1,p=[1-dom_prob,dom_prob]):   
                adj[:,int(end_node),path[0]] = list_adj[0][:,int(end_node),path[0]]   # add the first operation
                adj[:,path[0],int(end_node)] = list_adj[0][:,path[0],int(end_node)]   
                m = len(path)
                i = 0
                j = 1
                while (j<m):  # update the operation in the path
                    adj[:,path[i],path[j]] = list_adj[0][:,path[i],path[j]]
                    adj[:,path[j],path[i]] = list_adj[0][:,path[j],path[i]]
                    i+=1
                    j+=1
    # add the sub path
    for end_node in path_sub.keys():
        for path in path_sub[end_node]:
            if np.random.choice(2,1,p=[1-sub_prob,sub_prob]):  
                # if torch.count_nonzero(adj[:,int(end_node),path[0]]).item() == 0:
                adj[:,int(end_node),path[0]] = list_adj[1][:,int(end_node),path[0]]   # add the first operation
                adj[:,path[0],int(end_node)] = list_adj[1][:,path[0],int(end_node)]   
                m = len(path)
                i = 0
                j = 1
                while (j<m):  # update the operation in the path
                    # if torch.count_nonzero(adj[:,path[i],path[j]]).item() == 0:
                    adj[:,path[i],path[j]] = list_adj[1][:,path[i],path[j]]
                    adj[:,path[j],path[i]] = list_adj[1][:,path[j],path[i]]
                    i+=1
                    j+=1
    adj = mutate(adj,num_node=num_node,prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker)
    return adj

def cross_over2(adj0,adj1,dom_prob=0.7,cross_mutate_prob=0.08,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],num_node=13):
    index = np.random.choice(2,1)   # choose an architecture to be the domination
    list_adj = []
    if index==0:
        list_adj.append(adj0)
        list_adj.append(adj1)
    else:
        list_adj.append(adj1)
        list_adj.append(adj0) 
    adj = torch.zeros_like(adj0)
    # add the dominate path first
    adj[:,0,:] = list_adj[0][:,0,:]
    adj[:,:,0] = list_adj[0][:,:,0]
    adj[:,1,1:] = list_adj[0][:,1,1:]
    adj[:,1:,1] = list_adj[0][:,1:,1]
    for j in range(2,12):
        if np.random.choice(2,1,[1-dom_prob,dom_prob]):
            adj[:,j,j:] = list_adj[0][:,j,j:]
            adj[:,j:,j] = list_adj[0][:,j:,j]
        else:
            adj[:,j,j:] = list_adj[1][:,j,j:]
            adj[:,j:,j] = list_adj[1][:,j:,j]
    adj = mutate(adj,num_node=num_node,prob=cross_mutate_prob,conv_ker=conv_ker,avg_max_ker=avg_max_ker)
    return adj

def reverse_to_sigmoid_toan_bo(adj1,conv_ker=[1,3,5,9,19,39],avg_max_ker=[3,5,9],num_node=13):
    # this only for SETTING THRESHOLD so that adj can be convert to sigmoid form
    num_conv_ker = len(conv_ker)
    num_avgmax_ker = len(avg_max_ker)
    conv_threshold = []    # if pass a certain threshold then set the adj[i,j,k] = kernel respectively
    avg_max_threshold = []
    for i in range(num_conv_ker):  
        assert conv_ker[i]%2 ==1, "convolution kernel size must be an odd number!"
        if i != num_conv_ker-1:
            conv_threshold.append((np.log(conv_ker[i])+np.log(conv_ker[i+1]))/2)   # find the middle point
        else:
            conv_threshold.append(np.ceil(np.log(conv_ker[i])))  # the last thresholds is the ceil(log(kernel))
    for i in range(num_avgmax_ker):
        assert avg_max_ker[i]%2 ==1, "averagepool or maxpool kernel size must be an odd number!"
        if i != num_avgmax_ker-1:
            avg_max_threshold.append((np.log(avg_max_ker[i])+np.log(avg_max_ker[i+1]))/2)   # find the middle point
        else:
            # NOTE: the term "-0.5" is because I want the scale from range 0->2.5 so that it would not favor kernel_size=9 more than other kernelsizes
            avg_max_threshold.append(np.ceil(np.log(avg_max_ker[i]))-0.5)
            
    # convert adj[i,j,k] to the range of [0,1)
    adj = adj1.clone()
    for i in range(4):       
        for j in range(num_node):
            for k in range(num_node):
                if adj[i,j,k] != 0:
                    adj[i,j,k] = reverse_to_sigmoid(i,adj[i,j,k],conv_threshold,avg_max_threshold,conv_ker,avg_max_ker) if not _SE_ else reverse_to_sigmoid_SE(i,adj[i,j,k],conv_threshold,avg_max_threshold,conv_ker,avg_max_ker)
    return adj

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


# adj = torch.rand((4,13,13))
# adj1 = randomly_generate(0.2)
# model = Create_model(adj1,6,dropout=0.0)
# conv_ker = [i*2+1 for i in range(23)]
# avg_max_ker = [i*2+3 for i in range(4)]
# adj = randomly_generate(0.2)
# adj, _, _ = preprocess_adj_SE(adj)
# adj1 = reverse_to_sigmoid_toan_bo(adj)
# print(adj)
# adj1, _, _ = preprocess_adj(adj1)
# paths = extract_path(adj1)
# print(paths)
# new_adj = cross_over2(adj,adj1)
# print(new_adj)
# adj1 = finetune_NAS(adj)
# adj1, _, _ = preprocess_adj(adj1,conv_ker=conv_ker,avg_max_ker=avg_max_ker)
# print(adj1-adj)
# conv_ker = [i*2+1 for i in range(23)]
# avg_max_ker = [i*2+3 for i in range(5)]

# x = torch.rand((1,12,1000))
# y = model.forward(x)
# adj, _, _ =preprocess_adj(adj)
# adj1 = mutate(adj)
# model1 = Create_model(adj1,6,conv_ker=conv_ker,avg_max_ker=avg_max_ker)
# x = torch.rand((1,12,1000))
# y = model1.forward(x)

# adj1, _, _ = preprocess_adj(adj1)
# print((adj1-adj).sum())

# adj = torch.load("Best_adj6", map_location=torch.device('cpu'))
# adj = reverse_to_sigmoid_toan_bo(adj)
# model = Create_model(adj,6)
# print(get_n_params(model))