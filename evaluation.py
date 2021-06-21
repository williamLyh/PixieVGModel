from torchvision import models
import torchvision
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import numpy as np

import pickle
import gc
from tqdm import tqdm
import pandas as pd

def compute_variance_sum(preds,cov):
    return torch.sum(torch.mul(preds.T, torch.mm(cov,preds.T)),0)

def approx_E_log_sig(mean, cov, preds):
    # This is the approximation of E[log(Sigmoid(pred*pixie))]
    # mean and cov are the parameters of the q(s), estimated by inference network
    # preds has dimension N*Dim_pixie, where N is the number of predicate.
    a,b,c,d = 0.205, -0.319, 0.781, 0.870    
    temp = (torch.matmul(preds, mean) + b*(compute_variance_sum(preds,cov).pow(c)))/ \
           torch.sqrt(1+a*(compute_variance_sum(preds,cov).pow(d)))
    return torch.log(torch.sigmoid(temp))

def approx_E_sig(mean, cov, preds):
    a = 0.368
    temp = torch.matmul(preds, mean) / torch.sqrt(1+a*compute_variance_sum(preds,cov))
    return torch.sigmoid(temp)

def load_trained_model(model_path, device):
    [W_mu, W_cov] = pickle.load(open(model_path+"world_parameters.p", "rb"))
    V = pickle.load(open(model_path+"Lexical_parameters.p", "rb"))
    W_mu = W_mu.to(device)
    W_cov = W_cov.to(device)
    V = V.to(device)
    return W_mu, W_cov, V

def load_relpron(data_path):
    pixie_preds, test_pred, POS, vocab = [],[],[],[]
    line_num = 0
    with open(data_path+'RELPRON/relpron.all') as f:
        for line in f:
            line_num += 1
            words = line.split(' ')
            pred = words[1].split('_')[0]
            if words[0] == 'OBJ':
                subj = words[4].split('_')[0]
                verb = words[5].split('_')[0]
                obj = words[2].split('_')[0]
            else:
                subj = words[2].split('_')[0]
                verb = words[4].split('_')[0]
                obj = words[5].split('_')[0]
            
            # filtering word not in predicate list
            pixie_preds.append([subj,verb,obj])
            test_pred.append(pred)
            POS.append(2 if words[0]=='OBJ' else 0)
            vocab += [subj,verb,obj,pred]

    return pixie_preds, test_pred, POS, np.unique(vocab)

def load_men(data_path):
    path_men = data_path+'MEN/MEN_dataset_lemma_form.test'
    pixie_pred,test_pred,scores, POS = [], [], [],[]
    with open(path_men) as f:
        for line in f:
            word1 = line.split(' ')[0].split('-')[0]
            word2 = line.split(' ')[1].split('-')[0]
            pixie_pred.append(word1)
            pos_name = line.split(' ')[0].split('-')[1]
            if pos_name=='n':
                POS.append(0)
            elif pos_name=='v':
                POS.append(1)
            elif pos_name=='j':
                POS.append(2)
            else:
                print("false POS in MEN")
                assert False
            test_pred.append(word2)
            scores.append(float(line.split(' ')[2]))
    return pixie_pred, test_pred, POS, scores


def load_simlek999(data_path):
    path_simlex = data_path+'SimLex-999/SimLex-999.txt'
    df = pd.read_csv(path_simlex, delimiter = "\t")
    pixie_pred = df['word1'].values
    test_pred = df['word2'].values

    pos_name = df['POS'].values
    POS=[]
    for p in pos_name:
        if p=='N':
            POS.append(0)
        elif p=='V':
            POS.append(1)
        elif p=='A':
            POS.append(2)
        else:
            print("False POS for Simlek999 ")
    scores = df['SimLex999'].values

    return pixie_pred, test_pred, POS, scores


def load_gs11(data_path):
    path_gs11 = data_path+'GS2011data.txt'
    df = pd.read_csv(path_gs11, delimiter = " ")
    pixie_preds = df[['subject','verb','object']].values
    test_pred = df['landmark'].values
    scores = df['input'].values
    POS=[1]*len(pixie_preds)
    return pixie_preds, test_pred, POS, scores


class VariationalInferenceModel(nn.Module):
    def __init__(self, situation_dim):
        super(VariationalInferenceModel,self).__init__()
        self.q_mu = nn.Parameter(torch.randn(situation_dim))
        self.logCovDiag = nn.Parameter(torch.randn(situation_dim))
        
    def forward(self):
        q_cov = torch.diag(torch.exp(self.logCovDiag))
        return self.q_mu, q_cov

def generate_vocab(data_path):
    Y_flat = pickle.load(open(data_path+"y_preprocessed.p", "rb"))
    predicate_list, predicate_count = np.unique(np.array(Y_flat).reshape(-1),return_counts=True)
    predicates_table = {w:i for i,w in enumerate(predicate_list)}
    return predicate_list, predicates_table

def filter_data(pixie_pred, test_pred, scores, POS=None):
    def word2int(words,vocab_tab):
        return [vocab_tab[word] for word in words]

    predicate_list, predicates_table = generate_vocab(data_path+'pixie_data/data_pca/')
    flag = []
    for pixie, pred in zip(pixie_pred, test_pred):
        if (pixie in predicate_list) and (pred in predicate_list):
            flag.append(True)
        else:
            flag.append(False)
    flag = np.array(flag)

    covered_pixies = word2int(np.array(pixie_pred)[flag], predicates_table)
    covered_preds = word2int(np.array(test_pred)[flag], predicates_table)
    covered_POS = np.array(POS)[flag] if POS else None
    covered_scores = np.array(scores)[flag]
    print("vocab coverage {} our of {}".format(len(covered_pixies), len(pixie_pred)))

    return covered_pixies, covered_preds, covered_scores, covered_POS


def evaluate_men(data_path, device):
    pixie_pred, test_pred, POS, scores = load_men(data_path)
    covered_pixies, covered_preds, covered_scores, covered_POS = filter_data(pixie_pred,test_pred,scores, POS)
    _ = perform_variational_inference(device, covered_pixies, covered_preds, covered_scores, covered_POS)
    return covered_pixies, covered_preds, covered_POS, covered_scores

def perform_variational_inference(device, pixies, preds, scores, POS):
    print("Performing Variational Inference")
    pixie_dim, lr, dr, epoch_num = 20, 0.05, 0.0000000001, 400
    model_path = ''
    W_mu, W_cov, V = load_trained_model(model_path, device)
    loss_history_list = []
    pred_rank = []

    for p1,p2,pos in zip(pixies, preds, POS):
        model = VariationalInferenceModel(pixie_dim)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = dr)
        
        loss_history = []
        for epoch in range(epoch_num):
            # if np.mod(30,epoch)==0 & epoch!=0:
            #     lr = lr *0.3
            #     optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay = dr)

            q_mu, q_cov = model()
            optimizer.zero_grad()

            W_mu_pos = W_mu[pixie_dim*pos:pixie_dim*(pos+1)]
            W_cov_pos = W_cov[pixie_dim*pos:pixie_dim*(pos+1),pixie_dim*pos:pixie_dim*(pos+1)]

                
            Dqp = -torch.log(torch.det(q_cov))  \
                + torch.sum(torch.mul((q_mu-W_mu_pos).T, torch.matmul(torch.inverse(W_cov_pos),(q_mu-W_mu_pos).T))) \
                + torch.sum(torch.diagonal(torch.matmul(torch.inverse(W_cov_pos), q_cov),dim1=-1, dim2=-2))
            
            pred_loss = approx_E_log_sig(q_mu, q_cov, torch.unsqueeze(V[p1],0)) \
                        - torch.log(torch.sum(approx_E_sig(q_mu, q_cov, V)))

    #         pred_loss = torch.log(approx_E_sig(q_mu,q_cov, torch.unsqueeze(V[x1],0))) 
    #                     - torch.log(torch.sum(approx_E_sig(q_mu, q_cov, V)))
        
            loss = 0.1*Dqp - pred_loss
            
            loss.backward()
            optimizer.step()  
            loss_history.append(loss.item())
        
        loss_history_list.append(loss_history)
        # evaluation
        truth = approx_E_sig(q_mu, q_cov, torch.unsqueeze(V[p2],0)).item()
        
        sorted_truth,truth_index = torch.sort(approx_E_sig(q_mu, q_cov, V).cpu().detach(), descending=True)
        truth_rank = (truth_index == p2).nonzero().item()  # give rank index
        
        pred_rank.append(truth_rank)
        print(pred_rank)
        assert False
    return pred_rank

    

if __name__ == '__main__':
    data_path = "/local/scratch/yl535/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # X_eval, Y_eval, POS, scores = load_gs11(data_path)
    # X_eval, Y_eval, scores = load_gs11(data_path)

    # print(X_eval[0])
    # print(Y_eval[0])
    # print(np.unique(POS))
    # print(scores)

    # covered_pixies, covered_preds, covered_POS, covered_scores = evaluate_men(data_path)
    # print(covered_pixies, covered_preds,covered_scores)

    evaluate_men(data_path, device)