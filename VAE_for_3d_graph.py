import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_VAE(nn.Module):
    def __init__(self, number_of_nodes=13, input_dim=16, hidden_dim=16, latent_dim=16, num_hops=5, num_mlp_layers=1,
                 dropout=0.2, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.num_mlp_layers = num_mlp_layers

        self.init_weight = nn.parameter.Parameter(torch.rand(size=(number_of_nodes, input_dim)), requires_grad=True)
        self.eps = nn.parameter.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, number_of_nodes, input_dim, hidden_dim, hidden_dim, num_ops=number_of_nodes))
            else:
                self.mlps.append(MLP(num_mlp_layers, number_of_nodes, hidden_dim, hidden_dim, hidden_dim, num_ops=number_of_nodes))
            self.batch_norms.append(nn.BatchNorm1d(number_of_nodes))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim, dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu) # (128,13,latent_dim)
        else:
            return mu
        
    # NOTE: this assumes that the batchsize is 128  
    def _encoder(self, adj):
        # batch_size, _, node_num, _ = adj.shape # shape (128,4,13,13)
        x = torch.sum(torch.matmul(adj,self.init_weight),dim=1,keepdim=True) # shape (128,1,13,input_dim)
        list_adj = torch.split(adj,1,dim=1) # shape of each tensor (128,1,13,13)
        for l in range(self.num_layers - 1):
            list_h = torch.split(x,self.input_dim//4,dim=3) # shape of each tensor (128,1,13,input_dim//4 or hidden_dim//4)
            neighbor = []
            for i in range(4):
                neighbor.append(torch.matmul(list_adj[i], list_h[i])) # shape of each tensor (128,1,13,input_dim//4)
            neighbor = torch.concatenate(neighbor, dim=3) # shape (128,1,13,input_dim) or same as x
            agg = (1 + self.eps[l]) * x + neighbor 
            agg = agg.squeeze(dim=1)   # (128,13,input/hidden_dim)
            x = F.leaky_relu(self.batch_norms[l](self.mlps[l](agg)))# output of for-loop (128,1,13,output_dim)
            x = x.unsqueeze(dim=1)     # (128,1,13,input/hidden_dim)
        mu = torch.squeeze(self.fc1(x),dim=1)
        logvar = torch.squeeze(self.fc2(x),dim=1)
        return mu, logvar # (128,13,latent_dim)

    def forward(self, adj):
        mu, logvar = self._encoder(adj)
        z = self.reparameterize(mu, logvar)  # (128,13,latent_dim)
        adj_recon = self.decoder(z) 
        return adj_recon, mu, logvar
    
class Model_VAE_1(nn.Module):
    def __init__(self, number_of_nodes=13, input_dim=16, hidden_dim=16, latent_dim=16, num_hops=5, num_mlp_layers=1,
                 dropout=0.2, num_ops = 4, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.num_ops = num_ops

        self.init_weight = []
        for i in range(num_ops):
            self.init_weight.append(nn.parameter.Parameter(torch.rand(size=(number_of_nodes, input_dim)), requires_grad=True))
        self.init_weight = nn.ParameterList(self.init_weight)
        self.eps = nn.parameter.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, number_of_nodes, input_dim, hidden_dim, hidden_dim, num_ops))
            else:
                self.mlps.append(MLP(num_mlp_layers, number_of_nodes, hidden_dim, hidden_dim, hidden_dim, num_ops))
            self.batch_norms.append(nn.BatchNorm2d(num_ops))
        # convert to latent space
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.input_dim, dropout, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu) # (128,13,latent_dim)
        else:
            return mu
        
    # NOTE: this assumes that the batchsize is 128  
    def _encoder(self, adj):
        # batch_size, _, node_num, _ = adj.shape # shape (128,4,13,13)
        list_adj = torch.split(adj,1,dim=1) # shape of each tensor (128,1,13,13)
        x = torch.concat([torch.matmul(list_adj[i],self.init_weight[i]) for i in range(4)],dim=1) # shape (128,4,13,input_dim)
        for l in range(self.num_layers - 1):
            list_h = torch.split(x,1,dim=1) # shape of each tensor (128,1,13,input_dim or hidden_dim)
            neighbor = []
            for i in range(self.num_ops):
                neighbor.append(torch.matmul(list_adj[i], list_h[i])) # shape of each tensor (128,1,13,input_dim or hidden_dim)
            neighbor = torch.concatenate(neighbor, dim=1) # shape (128,4,13,input_dim or hidden dim)
            agg = (1 + self.eps[l]) * x + neighbor 
            # agg = agg.squeeze(dim=1)   # (128, 4, 13, input/hidden_dim)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)))  # (128,4,13,hidden dim or output dim)
        x = torch.mean(x, dim = 1, keepdim=True)    # (128,1,13,output dim)
        mu = torch.squeeze(self.fc1(x),dim=1)
        logvar = torch.squeeze(self.fc2(x),dim=1)
        return mu, logvar # (128,13,latent_dim)

    def forward(self, adj):
        mu, logvar = self._encoder(adj)
        z = self.reparameterize(mu, logvar)  # (128,13,latent_dim)
        adj_recon = self.decoder(z) 
        return adj_recon, mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout):
        super(Decoder, self).__init__()
        # self.activation_adj = activation_adj
        self.weight = nn.ModuleList()
        for _ in range(4):
            self.weight.append(torch.nn.Linear(embedding_dim, input_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedding):
        embedding = self.dropout(embedding)
        recon_adj = []
        for i in range(4):
            temp = self.weight[i](embedding)  # (128,13,latent_dim)
            recon_adj.append(F.relu(torch.matmul(temp, torch.permute(temp,(0,2,1)))))  # (128,13,13)
        return torch.stack(recon_adj,dim=1)   # (128,4,13,13) or same as adj
    
class MLP(nn.Module):
    def __init__(self, num_layers, number_of_nodes, input_dim, hidden_dim, output_dim, num_ops=4, **kwargs):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
            
        '''
        super(MLP, self).__init__()

        self.linear_or_not = False #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linears = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((num_ops)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.leaky_relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
        
class VAEReconstructed_L2_Loss(object):
    def __init__(self, gamma = 0.5, loss_adj=nn.MSELoss):
        super().__init__()
        self.gamma = gamma
        self.loss_adj = loss_adj()

    def __call__(self, inputs, mu, logvar, targets): # (128,4,13,13) and (128,13,latent_dim)
        loss_adj = self.loss_adj(inputs, targets)/4
        KLD = -0.5 / (inputs.shape[0] * inputs.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return self.gamma * loss_adj + (1-self.gamma) * KLD   
    
class Encoder_conv(nn.Module):
    def __init__(self,num_node,channel=48,act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(4,channel,(1,num_node))
        self.bn = nn.BatchNorm2d(channel)
        self.act = act()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
    def forward(self,adj):   # adj has the shape of (bs,4,13,13)
        z = self.conv(adj)   # shape (bs,channel,num_node=13,1)
        z = self.bn(z)
        z = self.act(z)  
        return self.flat(self.pool(z)), z   # shape (bs,channel) and (bs,channel,num_node,1)

class Decoder_conv(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.conv = nn.Conv2d(channel,4,1)
        self.act = nn.ReLU()
    def forward(self,z):    # shape (bs,c) 
        adj = torch.matmul(z,torch.transpose(z,3,2))  # (bs,channel,num_node,num_node)
        adj = self.conv(adj)    # (bs,4,num_node,num_node)
        adj = self.act(adj)
        return adj
class Autoencoder_conv(nn.Module):
    def __init__(self,num_node,channel=48,act=nn.ReLU):
        super().__init__()
        self.encoder = Encoder_conv(num_node,channel,act)
        self.decoder = Decoder_conv(channel=channel)
    def forward(self,adj):
        repre, z = self.encoder(adj)
        recon_adj = self.decoder(z)
        return recon_adj, z, repre
    
m = Model_VAE_1(13,16,16,16,3,2)
# x = torch.rand(1,4,13,13)
# # m.eval()
# y = list(m(x))
# targets = torch.rand(1,4,13,13)
# inputs, mu, logvar = y
# criterion = VAEReconstructed_L2_Loss()
# loss = criterion(inputs,mu,logvar,targets)
# loss.backward()