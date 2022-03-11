import argparse
from numpy.lib.function_base import cov
from torch.nn import parameter
from torchvision import models
import torchvision
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score
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

def load_men(data_path, vocab_path):
    path_men = data_path+'MEN/MEN_dataset_lemma_form_full'

    data = pd.DataFrame()
    predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))
    EVA_vocab_path = 'filtered_men.p'
    filtered_EVA_data = pickle.load(open(EVA_vocab_path, "rb"))
    filtered_EVA_data = [(pair[0].split('.')[0], pair[1].split('.')[0]) for pair in filtered_EVA_data]
    
    pixie_pred, test_pred, scores, pos = [], [], [],[]
    oov_flag, eva_flag = [], []

    with open(path_men) as f:
        for line in f:
            word1 = line.split(' ')[0].split('-')[0]
            word2 = line.split(' ')[1].split('-')[0]
            pixie_pred.append(word1)
            test_pred.append(word2)
            scores.append(float(line.split(' ')[2]))

            pos_name = line.split(' ')[0].split('-')[1]
            if pos_name=='n':
                pos.append(0)
            elif pos_name=='v':
                pos.append(1)
            elif pos_name=='j':
                pos.append(2)
            else:
                print("false POS in MEN")
                pos.append(3)

            # flags showing if the datapoint is oov
            if (word1 in predicate_list) and (word2 in predicate_list):
                oov_flag.append(0)
            else:
                oov_flag.append(1)
            
            # flags showing if the datapoint is EVA-oov
            if (word1, word2) in filtered_EVA_data:
                eva_flag.append(0)
            else:
                eva_flag.append(1)

    data['word1'], data['word2'], data['pos'], data['reference_scores'] = pixie_pred, test_pred, pos, scores
    data['oov'], data['eva_oov'] = oov_flag, eva_flag
    return data


def load_simlek999(data_path, vocab_path):
    path_simlex = data_path+'SimLex-999/SimLex-999.txt'
    simlek_df = pd.read_csv(path_simlex, delimiter = "\t")
    predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))

    EVA_vocab_path = 'filtered_simlek.p'
    filtered_EVA_data = pickle.load(open(EVA_vocab_path, "rb"))
    filtered_EVA_data = [(pair[0].split('.')[0], pair[1].split('.')[0]) for pair in filtered_EVA_data]

    oov_flag, eva_flag, pos = [], [], []
    for index, row in simlek_df.iterrows():
        if (row['word1'] in predicate_list) and (row['word2'] in predicate_list):
            oov_flag.append(0)
        else:
            oov_flag.append(1)

        if (row['word1'], row['word2']) in filtered_EVA_data:
            eva_flag.append(0)
        else:
            eva_flag.append(1)

        if row['POS']=='N':
            pos.append(0)
        elif row['POS']=='V':
            pos.append(1)
        elif row['POS']=='A':
            pos.append(2)
        else:
            print("False POS for Simlek999 ")
            pos.append(3)
        
    simlek_df['pos'] = pos
    simlek_df['oov'] = oov_flag
    simlek_df['eva_oov'] = eva_flag
    simlek_df['reference_scores'] = simlek_df['SimLex999']

    return simlek_df[['word1', 'word2', 'pos', 'reference_scores', 'oov', 'eva_oov']]


def load_gs11(data_path, vocab_path):
    path_gs11 = data_path+'GS2011data.txt'
    gs11_df = pd.read_csv(path_gs11, delimiter = " ")
    predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))

    EVA_vocab_path = 'filtered_gs11.p'
    filtered_EVA_data = pickle.load(open(EVA_vocab_path, "rb"))
    # filtered_EVA_data = [(row[0], row[1]) for row in filtered_EVA_data]
    filtered_EVA = []
    for i in range(len(filtered_EVA_data[0])):
        filtered_EVA.append((list(filtered_EVA_data[0][i]),filtered_EVA_data[1][i]))

    oov_flag, eva_flag = [], []
    for index, row in gs11_df.iterrows():
        if sum([word in predicate_list for word in row[['subject','verb','object','landmark']]]) == 4:
            oov_flag.append(0)
        else: 
            oov_flag.append(1)

        if (list(row[['subject','verb','object']]), row['landmark']) in filtered_EVA:
            eva_flag.append(0)
        else:
            eva_flag.append(1)

    gs11_df['word1'] = gs11_df[['subject','verb','object']].values.tolist()
    gs11_df['word2'] = gs11_df['landmark']
    gs11_df['pos'] = 1
    gs11_df['oov'] = oov_flag
    gs11_df['eva_oov'] = eva_flag
    gs11_df['reference_scores'] = gs11_df['input']

    return gs11_df[['word1', 'word2', 'pos', 'reference_scores', 'oov', 'eva_oov']]


def load_relpron(data_path, vocab_path):

    data = pd.DataFrame()
    predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))

    # EVA_vocab_path = 'filtered_relpron.p'
    # filtered_EVA_data = pickle.load(open(EVA_vocab_path, "rb"))
    # filtered_EVA_data = [(row[0], row[1]) for row in filtered_EVA_data]

    # pixie_preds, test_pred, head_noun_POS, vocab = [],[],[],[]
    # property_list, term_list = [], []
    # term_table = {}
    
    terms, triples, pos, oov_flag, ranks = [], [], [], [], [] # The 'rank' is the ranking within the term. 
    term_count = {}
    # line_num = 0
    # predicate_list, predicates_table = generate_vocab(vocab_path)
    # instance_cnt = 0
    with open(data_path+'RELPRON/relpron.all') as f:
        for line in f:
            # line_num += 1
            words = line.split(' ')
            term = words[1].split('_')[0]
            if words[0] == 'OBJ':
                subj = words[4].split('_')[0]
                verb = words[5].split('_')[0]
                obj = words[2].split('_')[0]
                head_noun_pos = 2
            else:
                subj = words[2].split('_')[0]
                verb = words[4].split('_')[0]
                obj = words[5].split('_')[0]
                head_noun_pos = 0
            
            terms.append(term)
            triples.append((subj, verb, obj))
            pos.append(head_noun_pos)
            if (term in predicate_list) and (subj in predicate_list) and (verb in predicate_list) and (obj in predicate_list):
                oov_flag.append(0)
            else:
                oov_flag.append(1)
            
            if term in term_count:
                term_count[term] +=1
            else:
                term_count[term] = 1
            ranks.append(term_count[term])

    data['term'] = terms
    data['triples'] = triples
    data['pos'] = pos
    data['ranks'] = ranks
    data['oov'] = oov_flag
    return data


            # # filtering word not in predicate list
            # if (term in predicate_list) and (subj in predicate_list) and (verb in predicate_list) and (obj in predicate_list):
            #     property_list.append([predicates_table[subj],
            #                             predicates_table[verb],
            #                             predicates_table[obj], head_noun_pos])
            #     if term in term_table:
            #         term_table[predicates_table[term]].append([predicates_table[subj],
            #                                                     predicates_table[verb],
            #                                                     predicates_table[obj], head_noun_pos])
            #     else:
            #         term_table[predicates_table[term]] =[[predicates_table[subj],
            #                                                 predicates_table[verb],
            #                                                 predicates_table[obj], head_noun_pos]]
            #         term_list.append(predicates_table[term])
            #     instance_cnt+=1

    return term_table, property_list, term_list


class VariationalInferenceModel(nn.Module):
    def __init__(self, situation_dim):
        super(VariationalInferenceModel,self).__init__()
        self.q_mu = nn.Parameter(torch.randn(situation_dim))
        self.logCovDiag = nn.Parameter(torch.randn(situation_dim))
        
    def forward(self):
        q_cov = torch.diag(torch.exp(self.logCovDiag))
        return self.q_mu, q_cov

def generate_vocab(data_path):
    # generate the vocabulary from the filtered pca transformed data.
    Y_flat = pickle.load(open(data_path+"y_preprocessed.p", "rb"))
    predicate_list, predicate_count = np.unique(np.array(Y_flat).reshape(-1),return_counts=True)
    predicates_table = {w:i for i,w in enumerate(predicate_list)}
    return predicate_list, predicates_table


def filter_evaluation_data(dataset, vocab_path, pixie_pred, test_pred, scores, POS=None, EVA_vocab_path=None):
    def word2int(words,vocab_tab):
        return [vocab_tab[word] for word in words]

    predicate_list, predicates_table = generate_vocab(vocab_path)
    if EVA_vocab_path != None:
        filtered_EVA_data = pickle.load(open(EVA_vocab_path, "rb"))
        flag = []
        flag_OOV=[]
        cnt_OOV=0
        for pixie, pred in zip(pixie_pred, test_pred):
            if (pixie in predicate_list) and (pred in predicate_list) and ((pixie+'.n',pred+'.n') in filtered_EVA_data):
                flag.append(True)
            else:
                flag.append(False)

            if ((pixie+'.n',pred+'.n') in filtered_EVA_data) and not ((pixie in predicate_list) and (pred in predicate_list)):
                flag_OOV.append(True)
                cnt_OOV+=1
            else:
                flag_OOV.append(False)
        print(cnt_OOV)
    else:
        flag = []
        for pixie,pred in zip(pixie_pred, test_pred):
            if dataset == 'GS2011':
                words = list(pixie)+[pred]
            else:
                words = [pixie, pred]
            # print(sum([w in predicate_list for w in words]))
            if sum([w in predicate_list for w in words])==len(words):
                flag.append(True)
            else:
                flag.append(False)

    # print(np.array(pixie_pred)[flag].shape)
    if dataset == 'GS2011':
        covered_pixies = [word2int(row,predicates_table) for row in np.array(pixie_pred)[flag]]
        # pickle.dump((np.array(pixie_pred)[flag],np.array(test_pred)[flag],np.array(POS)[flag],np.array(scores)[flag])  , open("filtered_gs11.p", "wb"))
        # print((np.array(pixie_pred)[flag],np.array(test_pred)[flag],np.array(POS)[flag],np.array(scores)[flag]))
        # print(np.array(pixie_pred)[flag].shape)
        # assert False
    else:
        covered_pixies = word2int(np.array(pixie_pred)[flag], predicates_table)
    print(covered_pixies)
    covered_preds = word2int(np.array(test_pred)[flag], predicates_table)
    covered_POS = np.array(POS)[flag] if POS else None
    covered_scores = np.array(scores)[flag]
    print("vocab coverage {} our of {}".format(len(covered_pixies), len(pixie_pred)))
    oov_scores = np.array(scores)[flag_OOV] if EVA_vocab_path != None else None

    return covered_pixies, covered_preds, covered_scores, covered_POS, oov_scores



def evaluate_dataset(dataset, evaluation_data_path, vocab_path, device, pixie_dim, model_path, EVA_vocab=False):
    def renormalize(pred, score):
        range1 = max(pred)-min(pred)
        range2 = max(score)- min(score)
        renormalized_pred = [round((val-min(pred))*range2/range1) + min(score) for val in pred]
        return renormalized_pred
        
    EVA_vocab_path = None   
    if dataset=='MEN':
        pixie_pred, test_pred, POS, scores = load_men(evaluation_data_path)
        EVA_vocab_path = 'filtered_men.p' if EVA_vocab else None
    elif dataset=='Simlek':
        pixie_pred, test_pred, POS, scores = load_simlek999(evaluation_data_path)
        EVA_vocab_path = 'filtered_simlek.p' if EVA_vocab else None
    elif dataset=='RELPRON':
        term_table, property_list, term_list = load_relpron(evaluation_data_path, vocab_path)
    elif dataset=='GS2011':
        pixie_pred, test_pred, POS, scores  = load_gs11(evaluation_data_path)
    else:
        print('wrong dataset')
    # assert False
    if dataset!='RELPRON':
        # relpron use different evaluation metrics
        covered_pixies, covered_preds, covered_scores, covered_POS, oov_scores = filter_evaluation_data(dataset, vocab_path, pixie_pred,test_pred,scores, POS, EVA_vocab_path)
        pred_rank, pred_truths = perform_variational_inference(device, covered_pixies, covered_preds, covered_scores, covered_POS,pixie_dim,model_path)
        if not EVA_vocab:
            renormalized_pred_rank = renormalize(pred_rank, covered_scores)
            correlation = stats.spearmanr(renormalized_pred_rank, covered_scores).correlation
            print("Spearman correlation truth: {}".format(stats.spearmanr(pred_truths, covered_scores).correlation))
            print("Spearman correlation score: {}".format(stats.spearmanr(pred_rank, covered_scores).correlation))
            print("Spearman renormalized correlation score: {}".format(correlation))
        else:
            oov_length = len(oov_scores)
            pred_rank += [np.median(pred_rank)]*oov_length
            pred_truths += [0.5]*oov_length
            covered_scores = np.concatenate([covered_scores,oov_scores])
            print(pred_rank,covered_scores)
            renormalized_pred_rank = renormalize(pred_rank, covered_scores)
            correlation = stats.spearmanr(renormalized_pred_rank, covered_scores).correlation
            pickle.dump((pred_rank,covered_scores), open('simlex_result.p','wb'))

            print("Spearman correlation truth: {}".format(stats.spearmanr(pred_truths, covered_scores).correlation))
            print("Spearman correlation score: {}".format(stats.spearmanr(pred_rank, covered_scores).correlation))
            print("Spearman renormalized correlation score: {}".format(correlation))
    else:
        # relpron use Mean Average Precision
        W_mu, W_cov, V = load_trained_model(model_path, device)

        buf = {}
        buf2 = {}
        for idx in tqdm(range(len(property_list))):
            property = property_list[idx]
            p1 = property[:3]
            pos = property[3]
            all_truths, loss_history = _single_variational_inference(p1,pos,W_mu, W_cov, V, device, pixie_dim)
            print(np.where(all_truths.numpy().argsort()[::-1]==p1[pos])[0].item())

            p_buf = {}
            for term in term_list:
                # rank = np.where(all_truths.numpy().argsort()[::-1]==term)[0].item()
                rank = _get_ranking(all_truths.numpy(),term_list,term)
                p_buf[term] = rank
                if term in buf2:
                    buf2[term][tuple(property_list[idx])] = rank
                else: 
                    buf2[term] = {tuple(property_list[idx]): rank}
            print(p_buf)
            buf[tuple(property_list[idx])]= p_buf

        APs = []
        for term, p_list in buf2.items():
            labels = [0]*len(property_list)
            pred_ranks = []
            for idx, (property, rank) in enumerate(p_list.items()):
                pred_ranks.append(rank)
                if list(property) in term_table[term]:
                    labels[idx] = 1
            APs.append(average_precision_score(labels, np.nan_to_num(pred_ranks,nan=max(pred_ranks))))
            print(APs)
        print("Mean Average Precision score: {}".format(np.mean(APs)))


def evaluate_MEN(evaluation_data_path, vocab_path, device, pixie_dim, parameter_path, use_EVA_vocab):
    data = load_men(evaluation_data_path, vocab_path)
    if use_EVA_vocab==0:
        print('vocab coverage: {} out of {}'.format(data.shape[0] - data.oov.sum(), data.shape[0]))
    else:
        print('EVA vocab is used. Vocab coverage: {} out of {}'.format(data.shape[0]- data.eva_oov.sum(), data.shape[0]))

    data = perform_variational_inference(data, device, pixie_dim, parameter_path, vocab_path, use_EVA_vocab)
    # data.to_csv('MEN_results')
    return data

def evaluate_simlek(evaluation_data_path, vocab_path, device, pixie_dim, parameter_path, use_EVA_vocab):
    data = load_simlek999(evaluation_data_path, vocab_path)
    if use_EVA_vocab==0:
        print('vocab coverage: {} out of {}'.format(data.shape[0] - data.oov.sum(), data.shape[0]))
    else:
        print('EVA vocab is used. Vocab coverage: {} out of {}'.format(data.shape[0]- data.eva_oov.sum(), data.shape[0]))

    data = perform_variational_inference(data, device, pixie_dim, parameter_path, vocab_path, use_EVA_vocab)
    # data.to_csv('MEN_results')
    return data

def evaluate_gs11(evaluation_data_path, vocab_path, device, pixie_dim, parameter_path, use_EVA_vocab):
    data = load_gs11(evaluation_data_path, vocab_path)
    if use_EVA_vocab==0:
        print('vocab coverage: {} out of {}'.format(data.shape[0] - data.oov.sum(), data.shape[0]))
    else:
        print('EVA vocab is used. Vocab coverage: {} out of {}'.format(data.shape[0]- data.eva_oov.sum(), data.shape[0]))

    data = perform_variational_inference(data, device, pixie_dim, parameter_path, vocab_path, use_EVA_vocab)
    # data.to_csv('MEN_results')
    return data

def evaluate_relpron(evaluation_data_path, vocab_path, device, pixie_dim, parameter_path):
    # contextual dataset always has loose filtering condition
    data = load_relpron(evaluation_data_path, vocab_path)
    # data = perform_variational_inference_relpron(data, device, pixie_dim, parameter_path, vocab_path)
    covered_data = data[data['oov']==0].reset_index()
    covered_data, truth_mat, rank_mat = perform_variational_inference_relpron(covered_data, device, pixie_dim, parameter_path, vocab_path)
    pickle.dump((covered_data, truth_mat, rank_mat), open('relpron_result.p','wb'))

    # truth_mat, rank_mat are len(property_list) * len(term_list)

    property_list = covered_data['triples'].to_list()
    term_list = list(covered_data['term'].unique())
    pos_list = covered_data['pos'].to_list()
    term_table = covered_data.groupby('term')['triples'].apply(list).to_dict()

    truth_APs = []
    rank_APs = []

    # for term, property_list in term_table.items():
    for j, term in enumerate(term_list):
        labels = [0]* len(property_list)
        # pred_ranks = []
        for i, property in enumerate(property_list):
            if property in term_table[term]:
                labels[i]=1
        truth_APs.append(average_precision_score(labels, truth_mat[:,j] ))
        rank_APs.append(average_precision_score(labels, rank_mat[:,j] ))
    print("Mean Average Precision score (truth): {}".format(np.mean(truth_APs)))

    print("Mean Average Precision score (rank): {}".format(np.mean(rank_APs)))



def perform_variational_inference_relpron(data, device, pixie_dim, parameter_path, vocab_path):
    def word2int(words):
        predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))
        if isinstance(words, str):
            return predicates_table[words]
        else:
            return [predicates_table[word] for word in list(words)]

    W_mu, W_cov, V = load_trained_model(parameter_path, device)

    term_list = list(data['term'].unique())
    property_list = data['triples'].to_list()

    truth_mat = np.zeros((len(property_list),len(term_list)))
    rank_mat = np.zeros((len(property_list),len(term_list)))



    # pred_truth, pred_rank = [],[]
    for i, row in data.iterrows():
        # if row['oov']==0:
        all_truths, loss_history = _single_variational_inference(word2int(row['triples']), row['pos'],W_mu,W_cov,V,device,pixie_dim)
        for j, term in enumerate(term_list):
            truth_rank = _get_ranking(all_truths.numpy(), word2int(term_list), word2int(term))
            # pred_truth.append(all_truths[word2int(term)].item())
            # pred_rank.append(truth_rank)
            truth_mat[i,j] = all_truths[word2int(term)].item()
            rank_mat[i,j] = truth_rank
        # else:
        #     pred_truth.append(-1)
        #     pred_rank.append(-1)


    # data['pred_truth'] = pred_truth
    # data['pred_rank'] = pred_rank
    return data, truth_mat, rank_mat


def perform_variational_inference(data, device, pixie_dim, parameter_path, vocab_path, use_EVA_vocab):
    def word2int(words):
        predicate_list, predicates_table = pickle.load(open(vocab_path+'vocabulary.p','rb'))
        if isinstance(words, str):
            return predicates_table[words]
        else:
            return [predicates_table[word] for word in list(words)]

    print("Performing Variational Inference")
    W_mu, W_cov, V = load_trained_model(parameter_path, device)

    pred_truth, pred_rank = [], []
    for index, row in data.iterrows():
        if  use_EVA_vocab==0:
            if row['oov']==0:
                all_truths, loss_history = _single_variational_inference(word2int(row['word1']), row['pos'],W_mu,W_cov,V,device,pixie_dim)
                # row['pred_truth'] = all_truths[word2int(row['word2'])]
                truth_rank = _get_ranking(all_truths.numpy(), word2int(data[data['oov']==0]['word2']), word2int(row['word2']))
                # row['pred_rank'] = truth_rank
                pred_truth.append(all_truths[word2int(row['word2'])].item())
                pred_rank.append(truth_rank)
            else:
                pred_truth.append(-1)
                pred_rank.append(-1)
        else:
            if row['oov']==0 and row['eva_oov']==0:
                all_truths, loss_history = _single_variational_inference(word2int(row['word1']), row['pos'],W_mu,W_cov,V,device,pixie_dim)
                # row['pred_truth'] = all_truths[word2int(row['word2'])]
                truth_rank = _get_ranking(all_truths.numpy(), word2int(data[data['oov']==0]['word2']), word2int(row['word2']))
                # row['pred_rank'] = truth_rank
                pred_truth.append(all_truths[word2int(row['word2'])].item())
                pred_rank.append(truth_rank)
            else:
                pred_truth.append(-1)
                pred_rank.append(-1)

    data['pred_truth'] = pred_truth
    data['pred_rank'] = pred_rank
    return data



def _get_ranking(all_truths, pred_list, pred):
    # return the ranking of truth of word2 in word2_list
    # Higher truth value, higher ranking
    sorted_truth = np.sort(all_truths)
    truth_index = np.argsort(all_truths)
    cnt=0
    for t, i in zip(sorted_truth,truth_index):
        if i in pred_list:
            cnt+=1
        if i ==pred:
            return cnt

def _single_variational_inference(pred,pos,W_mu, W_cov, V, device,pixie_dim):
    lr, epoch_num =  0.03,  800

    if not isinstance(pred,list):
        model = VariationalInferenceModel(pixie_dim)   # here depends on the dataset
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=lr,step_size_up=100,mode="triangular2",cycle_momentum=False)
        loss_history = []
        for epoch in range(epoch_num):
            q_mu, q_cov = model()
            optimizer.zero_grad()
            W_mu_pos = W_mu[pixie_dim*pos:pixie_dim*(pos+1)]
            W_cov_pos = W_cov[pixie_dim*pos:pixie_dim*(pos+1),pixie_dim*pos:pixie_dim*(pos+1)]
            Dqp = -torch.log(torch.det(q_cov))  \
                + torch.sum(torch.mul((q_mu-W_mu_pos).T, torch.matmul(torch.inverse(W_cov_pos),(q_mu-W_mu_pos).T))) \
                + torch.sum(torch.diagonal(torch.matmul(torch.inverse(W_cov_pos), q_cov),dim1=-1, dim2=-2))
            
            pred_loss = approx_E_log_sig(q_mu, q_cov, torch.unsqueeze(V[pred],0)) \
                        - torch.log(torch.sum(approx_E_sig(q_mu, q_cov, V)))
        
            loss = 0.07*Dqp - pred_loss
            loss.backward()
            optimizer.step()  
            scheduler.step()
            loss_history.append(loss.item())
        print('ELBO-prior: ', Dqp.item(), 'ELBO-likelihood: ', pred_loss.item())
        all_truths = approx_E_sig(q_mu, q_cov, V).cpu().detach()
        return all_truths, loss_history
    else:
        model = VariationalInferenceModel(pixie_dim*3)   # here depends on the dataset
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=lr,step_size_up=300,mode="triangular2",cycle_momentum=False)
        loss_history = []
        for epoch in range(epoch_num):
            q_mu, q_cov = model()
            optimizer.zero_grad()
            # W_mu_pos = W_mu[pixie_dim*pos:pixie_dim*(pos+1)]
            # W_cov_pos = W_cov[pixie_dim*pos:pixie_dim*(pos+1),pixie_dim*pos:pixie_dim*(pos+1)]
            Dqp = -torch.log(torch.det(q_cov)) \
                + torch.sum(torch.mul((q_mu-W_mu).T, torch.matmul(torch.inverse(W_cov),(q_mu-W_mu).T))) \
                + torch.sum(torch.diagonal(torch.matmul(torch.inverse(W_cov), q_cov),dim1=-1, dim2=-2))
            pred_loss = 0
            for i in range(len(pred)):
                q_mu_pos = q_mu[pixie_dim*i:pixie_dim*(i+1)]
                q_cov_pos = q_cov[pixie_dim*i:pixie_dim*(i+1),pixie_dim*i:pixie_dim*(i+1)]
                pred_loss += approx_E_log_sig(q_mu_pos, q_cov_pos, torch.unsqueeze(V[pred[i]],0)) \
                            - torch.log(torch.sum(approx_E_sig(q_mu_pos, q_cov_pos, V)))
        
            loss = 0.1*Dqp - pred_loss
            loss.backward()
            optimizer.step()  
            scheduler.step()
            loss_history.append(loss.item())
            # if epoch%500 == 0:
            #     print(Dqp.item(), pred_loss.item())
            #     print(torch.sum(torch.mul((q_mu-W_mu).T, torch.matmul(torch.inverse(W_cov),(q_mu-W_mu).T))).item())
        print('ELBO-prior: ', Dqp.item(), 'ELBO-likelihood: ', pred_loss.item())

        q_mu_pos = q_mu[pixie_dim*pos:pixie_dim*(pos+1)]
        q_cov_pos = q_cov[pixie_dim*pos:pixie_dim*(pos+1),pixie_dim*pos:pixie_dim*(pos+1)]
        all_truths = approx_E_sig(q_mu_pos, q_cov_pos, V).cpu().detach()
        return all_truths, loss_history
        

def calculate_spearman(data, use_eva_vocab):
    if use_eva_vocab:
        covered_data = data[data['eva_oov']==0]
    else:
        covered_data = data[data['oov']==0]

    truth_score = stats.spearmanr(covered_data['pred_truth'], covered_data['reference_scores']).correlation
    ranking_score = stats.spearmanr(covered_data['pred_rank'], covered_data['reference_scores']).correlation
    print("Spearman correlation truth score: {}".format(truth_score))
    print("Spearman correlation ranking score: {}".format(ranking_score))
        

if __name__ == '__main__':
    # Run the evaluation to generate prediction files.

    parser = argparse.ArgumentParser()
    parser.add_argument('--pixie_dim', type=int, default=100, help='dimension of pixie')
    # The datasets could be 'MEN', 'Simlek', 'RELPRON', 'GS2011'
    parser.add_argument('--dataset', type=str, default='MEN', help='evaluation dataset')
    # The path for evaluation datasets
    parser.add_argument('--evaluation_data_path', type=str, default='pixie_data/', help='path to save data')
    # This is the data you used to generate the vocabulary
    parser.add_argument('--pca_path', type=str, default='data_pca/', help='path to save PCA data')
    parser.add_argument('--parameter_path', type=str, default='parameters/', help='path to save parameters')
    parser.add_argument('--use_EVA_vocab', type=bool , default=False, help='if using EVA filtered vocab')

    args = parser.parse_args()

    # The path of evaluation datasets
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # evaluate_dataset(args.dataset, args.evaluation_data_path, args.pca_path, device, args.pixie_dim, args.parameter_path, args.use_EVA_vocab)

    if args.dataset == 'MEN':
        data = evaluate_MEN(args.evaluation_data_path, args.pca_path, device, args.pixie_dim, args.parameter_path, args.use_EVA_vocab)
        data.to_csv('MEN_result.csv')
        calculate_spearman(data, args.use_EVA_vocab)

    elif args.dataset == 'simlek':
        data = evaluate_simlek(args.evaluation_data_path, args.pca_path, device, args.pixie_dim, args.parameter_path, args.use_EVA_vocab)
        data.to_csv('simlek_result.csv')
        calculate_spearman(data, args.use_EVA_vocab)

    elif args.dataset == 'GS2011':
        data = evaluate_gs11(args.evaluation_data_path, args.pca_path, device, args.pixie_dim, args.parameter_path, args.use_EVA_vocab)
        data.to_csv('gs11_result.csv')
        calculate_spearman(data, args.use_EVA_vocab)

    elif args.dataset == 'relpron':
        # data = load_relpron(args.evaluation_data_path, args.pca_path)
        evaluate_relpron(args.evaluation_data_path, args.pca_path, device, args.pixie_dim, args.parameter_path)
        pass