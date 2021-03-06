import argparse
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import numpy as np
import os

import pickle
from tqdm import tqdm

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
    
class LossForLexiconModel(nn.Module):
    def __init__(self, mode='ML', lr=0.01, dr=5e-8, epoch_num=30):
        super(LossForLexiconModel,self).__init__()
        self.BCEcriterion = nn.BCELoss(reduction='mean')
        # best dr for ML is 5e-9
        self.mode = mode
        self.lr = lr
        self.dr = dr
        self.epoch_num = epoch_num 

    def forward(self, truth, prob, all_truth, y_target):
        if self.mode == 'BCE':
            y_target_flat = torch.zeros_like(all_truth)
            for idx,pos in enumerate(y_target):
                y_target_flat[idx,pos]=1    
            pred_loss = self.BCEcriterion(all_truth, y_target_flat)
        elif self.mode =='ML':
            # print(torch.sum(-torch.log(truth+0.01)).item(),torch.sum(- torch.log(prob+0.01)).item())
            # pred_loss = torch.sum(-torch.log(truth))*0.5 + torch.sum(- torch.log(prob))
            pred_loss=torch.sum(- torch.log(prob))
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
def train_world_model(pixie_dim, num_semroles, data_path, parameter_path):
    world_model = WorldModel(pixie_dim, num_semroles)
    x = pickle.load(open(data_path+"x_preprocessed.p", "rb"))
    mu_batch, cov_batch = world_model.estimate_parameters(torch.Tensor(x).reshape(-1,3*pixie_dim))
    world_model.modify_conditional_independecy()

    pickle.dump([world_model.W_mu.cpu().detach(), world_model.W_cov.cpu().detach()],
                 open(parameter_path+"world_parameters.p", "wb"))

# train Lexicon model:
def evluate_lexicon_model(test_loader, lexmodel, loss_func, loss_mode='ML'):
    lexmodel.eval()
    loss_all=0
    # loss_func = LossForLexiconModel(loss_mode)

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

def train_lexicon_model(pixie_dim, device, data_path, parameter_path, lr, dr, epoch_num, loss_mode='ML'):
    predicate_list, predicates_table = generate_vocab(data_path)
    predicate_size = len(predicate_list)

    X = pickle.load(open(data_path+"x_preprocessed.p", "rb")).reshape(-1,pixie_dim)
    Y = pickle.load(open(data_path+"y_preprocessed.p", "rb")).reshape(-1)
    r = torch.randperm(X.shape[0])    
    split_ratio = 0.9
    split_pos = int(X.shape[0]*split_ratio)
    X_train, Y_train = X[r][:split_pos], Y[r][:split_pos]
    X_dev, Y_dev = X[r][split_pos:], Y[r][split_pos:]

    train_dataset = LexiconDataset(X_train,Y_train,predicates_table)
    valid_dataset = LexiconDataset(X_dev,Y_dev,predicates_table)

    batch_size = 1024
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True) 

    loss_func = LossForLexiconModel(loss_mode, lr, dr, epoch_num)

    lexmodel = LexiconModel(pixie_dim, predicate_size)
    lexmodel.to(device)

    optimizer = torch.optim.Adam(lexmodel.parameters(), lr=loss_func.lr, weight_decay = loss_func.dr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.3)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=loss_func.lr,step_size_up=6,mode="triangular2",cycle_momentum=False)

    for epoch in range(loss_func.epoch_num):
        lexmodel.train()
        train_loss_all = 0
        word_learnt = []
        for batch in tqdm(train_loader):
            x_feature = batch['x_feature']
            y_label = batch['label']
            optimizer.zero_grad()
            truth, truth_all, prob = lexmodel(x_feature.to(device), y_label.to(device))
            pred_loss = loss_func(truth,prob,truth_all,y_label.to(device))
            pred_loss.backward()
            optimizer.step()  
            train_loss_all += pred_loss.item()
            word_learnt.append(torch.unique(y_label[prob>0.1]))
        print('words learnt ',len(torch.unique(torch.cat(word_learnt))))

        train_loss_average = train_loss_all/len(train_loader)
        scheduler.step()

        lexmodel.eval()
        valid_loss_average = evluate_lexicon_model(valid_loader, lexmodel, loss_func, loss_mode)
        print("==========================================")
        print("epoch: ", epoch+1)
        print("lr: ", optimizer.param_groups[0]["lr"])
        print("train loss:", train_loss_average)
        print("validation loss: ",valid_loss_average)

    pickle.dump(lexmodel.V.cpu().detach(), open(parameter_path+"Lexical_parameters.p", "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pixie_dim', type=int, default=100, help='dimension of pixie')
    parser.add_argument('--parameter_path', type=str, default='parameters/', help='path to save parameters')
    parser.add_argument('--pca_path', type=str, default='pca_data/', help='path to save PCA data')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dr', type=float, default=5e-8, help='decay rate')
    parser.add_argument('--epoch_num', type=int, default=20, help='number of epoch')
    args = parser.parse_args()

    # work_path = '/local/scratch/yl535/'
    # data_path = work_path + args.data_path + args.pca_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pixie_dim = args.pixie_dim
    if not os.path.isdir(args.parameter_path): os.mkdir(args.parameter_path)

    num_semroles = 2
    train_world_model(pixie_dim, num_semroles, args.pca_path, args.parameter_path)    
    train_lexicon_model(pixie_dim, device, args.pca_path, args.parameter_path, args.lr, args.dr, args.epoch_num)
