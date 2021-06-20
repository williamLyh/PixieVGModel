from torchvision import models
import torchvision
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import numpy as np

import pickle
import gc
from tqdm import tqdm
import pandas as pd

class LexiconDataset(Dataset):
    def __init__(self,image_features,predicates,predicates_table):
        # super(Dataset,self).__init__()
        self.x_features = image_features
        self.labels = np.array([predicates_table[y_pred] for y_pred in predicates])
        
    def __len__(self):
        return self.x_features.shape[0]

    def __getitem__(self,idx):
        
        datum={"x_feature": self.x_features[idx],
               "label": self.labels[idx]}

        return datum



class WorldModel(nn.Module):
    def __init__(self, pixie_dim, num_semroles):
        super(WorldModel,self).__init__()
        # semantic roles are fixed to 2 here, what if some data has only arg1, some has arg1&arg2 ?
        self.pixie_dim = pixie_dim
        self.num_pixie = num_semroles+1
        self.W_mu = torch.zeros(3*pixie_dim)
        self.W_cov = torch.zeros((3*pixie_dim, 3*pixie_dim))
        self.data_size = 0

    def forward(self, x):
        return x
    
    def estimate_parameters(self, x):
        batch_size = x.shape[0]
        mu_batch = torch.mean(x,0) 
        diff = x-mu_batch
        cov_batch = torch.matmul(diff.T,diff)/batch_size
        
        self.W_mu = self.W_mu*(self.data_size/(self.data_size+batch_size)) + mu_batch*(batch_size/(self.data_size+batch_size))
        self.W_cov = self.W_cov*(self.data_size/(self.data_size+batch_size)) + cov_batch*(batch_size/(self.data_size+batch_size))
        self.data_size += batch_size

        return mu_batch, cov_batch
    
    def modify_conditional_independecy(self):
        P = torch.inverse(self.W_cov)
        P[-self.pixie_dim:,:self.pixie_dim]=0
        P[:self.pixie_dim,-self.pixie_dim:]=0
        self.W_cov = torch.inverse(P)
    
    def inverse_cov(self):
        self.W_precision = torch.inverse(self.W_cov)


class LexiconModel(nn.Module):
    def __init__(self, pixie_dim, predicate_size):
        super(LexiconModel,self).__init__()
        self.predicate_size = predicate_size
        self.V = torch.nn.Parameter(torch.randn(predicate_size, pixie_dim))

    def forward(self, x, pred):
        truth, prob,truth_all = self.truth_and_prob(pred, x)
        return truth, truth_all, prob

    def truth(self, pred, pixie):
        v = self.V[pred,:]
        neg_energy = torch.sum(pixie*v,1)
        return torch.sigmoid(neg_energy)
    
    def all_truth(self, pixie):
        return torch.sigmoid(torch.matmul(pixie, self.V.t()))

    def truth_and_prob(self, pred, pixie):
        truth = self.truth(pred, pixie)
        truth_all = self.all_truth(pixie)
        prob = torch.div(truth, torch.sum(truth_all,1))
        return truth, prob,truth_all
    
class BCELossForLexiconModel(nn.Module):
    def __init__(self,mode='BCE'):
        super(BCELossForLexiconModel,self).__init__()
        self.BCEcriterion = nn.BCELoss(reduction='mean')
        self.mode = mode

    def forward(self,truth,prob, all_truth, y_target):
        y_target_flat = torch.zeros_like(all_truth)
        for idx,pos in enumerate(y_target):
            y_target_flat[idx,pos]=1    

        if self.mode == 'BCE':
            pred_loss = self.BCEcriterion(all_truth, y_target_flat)
        elif self.mode =='ML':
            pred_loss = torch.sum(-torch.log(truth))*1.5 + torch.sum(- torch.log(prob))
            pred_loss = pred_loss/y_target.shape[0]
        else:
            print("Loss function unrecognized")
        return pred_loss


def generate_vocab(data_path):
    Y_flat = pickle.load(open(data_path+"y_preprocessed.p", "rb"))
    predicate_list, predicate_count = np.unique(np.array(Y_flat).reshape(-1),return_counts=True)
    predicates_table = {w:i for i,w in enumerate(predicate_list)}
    return predicate_list, predicates_table

# train world model:
def train_world_model(pixie_dim, num_semroles,data_path):
    world_model = WorldModel(pixie_dim, num_semroles)
    x = pickle.load(open(data_path+"x_preprocessed.p", "rb"))
    mu_batch, cov_batch = world_model.estimate_parameters(torch.Tensor(x).reshape(-1,3*pixie_dim))
    world_model.modify_conditional_independecy(pixie_dim)

    pickle.dump([world_model.W_mu.cpu().detach(), world_model.W_cov.cpu().detach()],
                 open("world_parameters.p", "wb"))

# train Lexicon model:
def evluate_lexicon_model(test_loader, lexmodel):
    lexmodel.eval()
    loss_all=0
    loss_func = BCELossForLexiconModel()

    # data_size = 0
    for idx, test_batch in enumerate(test_loader):
        x_feature = test_batch['x_feature']
        y_label = test_batch['label']
        truth, truth_all, prob = lexmodel(x_feature.to(device), y_label.to(device))
        pred_loss = loss_func(truth,prob,truth_all,y_label.to(device))

        loss_all += pred_loss.item()
        # data_size += x_feature.shape[0]
    loss_average = loss_all/len(test_loader)
    return loss_average

def train_lexicon_model(pixie_dim, device, data_path):
    predicate_list, predicates_table = generate_vocab(data_path)
    predicate_size = len(predicate_list)

    X = pickle.load(open(data_path+"x_preprocessed.p", "rb")).reshape(-1,pixie_dim)
    Y = pickle.load(open(data_path+"y_preprocessed.p", "rb")).reshape(-1)
    r=torch.randperm(X.shape[0])    
    split_pos = int(X.shape[0]*0.8)
    X_train = X[r][:split_pos]
    Y_train = Y[r][:split_pos]
    X_dev = X[r][split_pos:]
    Y_dev = Y[r][split_pos:]
    train_dataset = LexiconDataset(X_train,Y_train,predicates_table)
    valid_dataset = LexiconDataset(X_dev,Y_dev,predicates_table)

    batch_size = 1000
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True) 

    learning_rate = 0.01
    decay_rate = 0.00000000001
    epoch_num = 30
    loss_func = BCELossForLexiconModel()

    lexmodel = LexiconModel(pixie_dim, predicate_size)
    lexmodel.to(device)

    optimizer = torch.optim.Adam(lexmodel.parameters(), lr=learning_rate, weight_decay = decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(epoch_num):
        lexmodel.train()
        train_loss_all = 0
        for batch in tqdm(train_loader):
            x_feature = batch['x_feature']
            y_label = batch['label']
            optimizer.zero_grad()
            truth, truth_all, prob = lexmodel(x_feature.to(device), y_label.to(device))

            pred_loss = loss_func(truth,prob,truth_all,y_label.to(device))
            # pred_loss = torch.sum(-torch.log(truth))*1.5 + torch.sum(- torch.log(prob))
            # # print(torch.sum(-torch.log(truth)).item(),torch.sum(- torch.log(prob)).item())

            # pred_loss = pred_loss/x_feature.shape[0]
#             pred_loss = torch.sum( - torch.log(prob+0.01))

            pred_loss.backward()
            optimizer.step()  
            train_loss_all += pred_loss.item()
        train_loss_average = train_loss_all/len(train_loader)
        scheduler.step()

        lexmodel.eval()
        valid_loss_average = evluate_lexicon_model(valid_loader, lexmodel)
        print("==========================================")
        print("epoch: ", epoch+1)
        print("lr: ", scheduler.get_lr())
        print("train loss:", train_loss_average)
        print("validation loss: ",valid_loss_average)

    pickle.dump(lexmodel.V.cpu().detach(), open("Lexical_parameters.p", "wb"))


if __name__ == '__main__':
    work_path = '/local/scratch/yl535/'
    data_path = work_path+'pixie_data/data_pca/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pixie_dim = 20
    num_semroles = 2
    # train_world_model(pixie_dim, num_semroles,data_path)
    # predicate_list, predicates_table = generate_vocab(data_path)
    train_lexicon_model(pixie_dim, device, data_path)

