
import matplotlib
import pandas as pd
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches
from torch.utils import data
from torch.utils.data import TensorDataset,DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchvision import models

from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.preprocessing import StandardScaler,normalize
import sklearn
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import numpy as np
import pickle
from tqdm import tqdm


def load_VG_data(VG_path):
    relations = pd.read_json(VG_path+'relationships.json')
    return relations

def features_extraction(relations, VG_path, save_path):
    def crop_image(img,obj):
        return img[obj['y']:obj['y']+obj['h'],
                   obj['x']:obj['x']+obj['w'],:]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    resnet = models.resnet101(pretrained=True)
    resnet.to(device)
    resnet.eval()
    Image_batch_size = 20000
    data_cut = 3000

    relations_len = relations.shape[0]
    ## corp and transform images:
    for file_cnt, relations_start_point in enumerate(range(0,relations_len, data_cut)):
        relations_end_point = min(relations_start_point+data_cut, relations_len)
        X,Y = [],[]
        for i in tqdm(range(relations_start_point,relations_end_point)):

            img_id = relations.iloc[i]['image_id']
            path_folder1 = VG_path+'VG_100K/{}.jpg'.format(img_id)
            path_folder2 = VG_path+'VG_100K_2/{}.jpg'.format(img_id)
            path = path_folder1 if os.path.isfile(path_folder1) else path_folder2

            img_pred = mpimg.imread(path)
            if len(img_pred.shape)<3:
                continue

            for relation in relations.iloc[i]['relationships']:
                if relation['synsets']==[]:
                    if relation['predicate']=='':
                        continue
                    else:
                        name_pred = [relation['predicate']]
                else:
                    name_pred = [term.split('.')[0] for term in relation['synsets']]
                    
                #### subj
                if relation['subject']['synsets']==[]:
                    subj = relation['subject']['name'] if 'name' in relation['subject'] else relation['subject']['names'][0]
                    if subj=='':
                        continue
                    else:
                        subjs = subj.split(' ')
                        name_subj = []
                        for term in subjs:
                            if wordnet.synsets(term)!=[]:
                                name_subj.append(wordnet.synsets(term)[0].name().split('.')[0])
                            else:
                                continue
                        if len(name_subj)==0:
                            continue
                else:
                    name_subj = [term.split('.')[0] for term in relation['subject']['synsets']]
                    
                #### obj
                if relation['object']['synsets']==[]:
                    obj = relation['object']['name'] if 'name' in relation['object'] else relation['object']['names'][0]
                    if obj=='':
                        continue
                    else:
                        objs = obj.split(' ')
                        name_obj = []
                        for term in objs:
                            if wordnet.synsets(term)!=[]:
                                name_obj.append(wordnet.synsets(term)[0].name().split('.')[0])
                            else:
                                continue
                        if len(name_obj)==0:
                            continue
                else:
                    name_obj = [term.split('.')[0] for term in relation['object']['synsets']]
                    
                img_subj = crop_image(img_pred,relation['subject'])
                img_obj = crop_image(img_pred,relation['object'])

                if (0 in img_subj.shape) or (0 in img_obj.shape):
                    continue
                for s in name_subj:
                    for p in name_pred:
                        for o in name_obj:
                            X.append([img_pred, img_subj, img_obj])
                            Y.append([p, s, o])

        print('X length:', len(X))

        X_batch = []
        for start_point in tqdm(range(0,len(X),Image_batch_size)):
            end_point = start_point+Image_batch_size if start_point+Image_batch_size<len(X) else len(X) 

            X_transformed = []
            print('image transforming')
            for row in X[start_point:end_point]:
                X_transformed += [transform(img) for img in row]
            X_transformed = torch.cat(X_transformed).reshape((-1,3,224,224)) # the X_transformed is 10000*3*224*224

            x_loader = DataLoader(X_transformed,batch_size=500)
            del X_transformed
            with torch.no_grad():     # have to use no_grad() otherwise there will be memory leck
                output = []
                print('passing CNN')
                for x_batch in x_loader:
                    output_batch = resnet(x_batch.to(device))
                    output.append(output_batch)
                    
            del x_loader
            X_batch.append(torch.cat(output).cpu().reshape((-1,3,1000)))
            del output

        print('saving data')
        pickle.dump(torch.Tensor(np.concatenate(X_batch)), open(save_path+"x_{}.p".format(file_cnt), "wb"))
        pickle.dump(Y, open(save_path+"y_{}.p".format(file_cnt), "wb"))
        del X_batch
        # del Y_flat

def generate_vocab(path):
    predicate_list = []
    batch_number = 37
    for itr in range(batch_number):
        Y_flat = pickle.load(open(path+"pixie_data/y_{}.p".format(itr), "rb"))
        predicate_list.append(np.array(Y_flat).reshape(-1))
        
    predicate_list, predicate_count = np.unique(np.concatenate(predicate_list),return_counts=True)
    predicates_table = {w:i for i,w in enumerate(predicate_list)}
    
    return predicate_list, predicates_table

def features_PCA(work_path,pixie_dim_new):
    pixie_dim = 1000
    cov_global = torch.zeros(pixie_dim,pixie_dim)
    data_size = 0
    batch_number = 37

    for itr in tqdm(range(batch_number)):
        x_hidden = pickle.load(open(work_path+"pixie_data/x_{}.p".format(itr), "rb")).reshape(-1, pixie_dim)
        diff = x_hidden - torch.mean(x_hidden,0)
        cov_batch = torch.matmul(diff.T,diff)/x_hidden.shape[0]
        cov_global = cov_global*(data_size/(data_size+x_hidden.shape[0])) \
                    + cov_batch*(x_hidden.shape[0]/(data_size+x_hidden.shape[0]))
        
        data_size += x_hidden.shape[0]
        
    # PCA transform matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_global)
    eigen_vecs_sorted = np.array([x for _, x in sorted(zip(eigen_vals, eigen_vecs), key=lambda pair: pair[0], reverse=True)])
    eigen_vals_sorted = sorted(eigen_vals, reverse=True)
    x_eigenval = eigen_vals_sorted[:pixie_dim_new]
    PCA_transform_matrix = eigen_vecs_sorted[:,:pixie_dim_new]

    # PCA transforming 
    x_batch = []
    y_batch = []
    # batch_cnt = 0
    for itr in tqdm(range(batch_number)):
        x_hidden = pickle.load(open(work_path+"pixie_data/x_{}.p".format(itr), "rb")).reshape(-1, pixie_dim)
        Y_flat = pickle.load(open(work_path+"pixie_data/y_{}.p".format(itr), "rb"))

        x_pca = x_hidden.reshape(-1,pixie_dim).numpy().dot(PCA_transform_matrix)
        x = x_pca/(0.4*np.sqrt(x_eigenval[:pixie_dim_new]))
        x_batch.append(torch.Tensor(x))
        y_batch.append(Y_flat)

        # if (np.mod(itr,20)==0 and itr!=0) or itr == batch_number-1:
    x_batch = torch.cat(x_batch)
    y_batch = np.concatenate(y_batch)
    
    pickle.dump(x_batch.reshape(-1,3,pixie_dim_new), open(work_path+"pixie_data/data_pca/x_preprocessed.p", "wb"))
    pickle.dump(y_batch, open(work_path+"pixie_data/data_pca/y_preprocessed.p", "wb"))
            
    # batch_cnt += 1

def plot_example(idx, re_idx,work_path, relations):
    img_id = relations.iloc[idx]['image_id']
    re = relations.iloc[idx]['relationships'][re_idx]
    path = work_path+'/visualgeno/VG_100K/{}.jpg'.format(img_id)
    path = path if os.path.isfile(path) else '../visualgeno/VG_100K_2/{}.jpg'.format(img_id)
    img = mpimg.imread(path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    pat_subject = patches.Rectangle((re['subject']['x'],re['subject']['y']),re['subject']['w'],re['subject']['h'],
                            lw=5, color='y',
                            fill=False)
    pat_object = patches.Rectangle((re['object']['x'],re['object']['y']),re['object']['w'],re['object']['h'],
                            lw=5, color='y',
                            fill=False)
    ax.add_patch(pat_subject)
    ax.add_patch(pat_object)
    plt.show(block=True)
    
    name_pred = re['predicate']
    name_subj = re['subject']['name'] if 'name' in re['subject'] else re['subject']['names']
    name_obj = re['object']['name'] if 'name' in re['object'] else re['object']['names']
    print('subject: ',name_subj)
    print('object: ',name_obj)
    print('relation: ',name_pred)

    # plot_example(4,8, relations)

if __name__ == '__main__':
    print('Processing the Visual Genome data')
    VG_path = '/local/scratch/yl535/visualgeno/'
    work_path = '/local/scratch/yl535/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    relations = load_VG_data(VG_path)
    features_extraction(relations,VG_path,save_path=work_path+'pixie_data')
    # plot_example(4,8,work_path, relations)
    # predicate_list, predicates_table = generate_vocab(work_path)
    
    features_PCA(work_path,pixie_dim_new=20)
