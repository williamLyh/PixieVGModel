from types import prepare_class
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

import pickle
import gc
from tqdm import tqdm
import pandas as pd

from pingouin import multivariate_normality
from scipy import stats
from scipy.stats import shapiro
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter



def load_trained_model(model_path, device):
    [W_mu, W_cov] = pickle.load(open(model_path+"world_parameters.p", "rb"))
    V = pickle.load(open(model_path+"Lexical_parameters.p", "rb"))
    W_mu = W_mu.to(device)
    W_cov = W_cov.to(device)
    V = V.to(device)
    return W_mu, W_cov, V

def load_data(data_path):
    X = pickle.load(open(data_path+"x_preprocessed.p", "rb"))
    Y = pickle.load(open(data_path+"y_preprocessed.p", "rb"))
    return X,Y


def direct_evaluate_world_model():
    pass

def direct_evaluate_lexicon_model():

    pass

if __name__ == '__main__':
    data_path = '/local/scratch/yl535/pixie_data/data_pca/'

    # X,Y = load_data(data_path)
    writer = SummaryWriter()

    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)