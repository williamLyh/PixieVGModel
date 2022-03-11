import pandas as pd
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchtext.data.utils import get_tokenizer
from torchvision import models

import numpy as np
import pickle
from tqdm import tqdm
import json
import zipfile
from collections import Counter
import argparse



def load_VG_data(rel_path):
    relations = pd.read_json(rel_path)
    return relations

def load_VG_obj(obj_path):
    objects = pd.read_json(obj_path)
    return objects

def clean_string(s):
    return s.strip().replace(' ','|').lower()


def generate_objects_vocab(obj_path, min_freq=30):
    print('generating object filtering checklist')
    all_objects={}
    zfile = zipfile.ZipFile(obj_path)
    for finfo in zfile.infolist():
        ifile=zfile.open(finfo)
        data = json.loads(ifile.read().decode('utf-8'))
        for image in data:
            for obj in image['objects']:
                objects = obj['synsets']
                objects_id = obj['object_id']
                # There could be more than 1 objet referring to 1 object_id 
                all_objects[objects_id] = [item[:-3] for item in objects]

    # some object_id refers to empty value, filter them:
    clean_all_objects = {}
    for obj_id, obj in all_objects.items():
        if obj != []:
            clean_all_objects[obj_id] = obj 

    # creat objects checklist by setting a min freq threshold
    objects_list = np.concatenate(list(clean_all_objects.values()))
    object_counts = Counter(objects_list)
    filtered_objects_counts = {}
    for object, count in object_counts.items():
        if count>min_freq:
            filtered_objects_counts[object] = count

    return filtered_objects_counts, clean_all_objects

def generate_rels_checklist(rel_path, filtered_objects_list, all_objects, min_freq=30):
    print('generating relation filtering checklist')
    relation_tokens = []
    zfile = zipfile.ZipFile(rel_path)
    unknown_failure_cnt=0
    key_absent_cnt = 0
    relation_cnt = 0
    all_objects_keys = set(all_objects.keys())
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        data = json.loads(ifile.read().decode('utf-8'))
        for rels in tqdm(data):
            for rel in rels['relationships']:
                # count single relation token 
                if len(rel['predicate']) > 0:
                    pred = clean_string(rel['predicate'])
                else:
                    continue

                subj_id = rel['subject']['object_id']
                try:
                    # Problems of preprocessing in EVA
                    # 1) filter out if not in all_objects
                    # some rels have synset, but the object_id is not in all_objects
                    # They are also filtered in EVA.
                    # The number could be around 0.4M
                    # 2) Sometimes all_objects contains the key, but has empty list 
                    for subj in all_objects[subj_id]:
                        if subj in filtered_objects_list:
                            relation_tokens.append((subj,pred))
                    relation_cnt += 1
                except:
                    if subj_id not in all_objects_keys:
                        key_absent_cnt +=1
                    else:
                        unknown_failure_cnt+=1
                        print(rel)
                    pass

                obj_id = rel['object']['object_id']
                try:
                    for obj in all_objects[obj_id]:
                        if obj in filtered_objects_list:
                            relation_tokens.append((pred,obj))
                    relation_cnt += 1
                except:
                    if obj_id not in all_objects_keys:
                        key_absent_cnt +=1
                    else:
                        unknown_failure_cnt+=1
                        print(rel)
                    pass
    print('unknown_failure_cnt: ', unknown_failure_cnt)
    print('key_absent_cnt: ', key_absent_cnt)
    print('include_relation_cnt: ', relation_cnt)

    # filter relations by frequency
    rels_counts = Counter(relation_tokens)
    rels_checklist = []
    for rel_token, count in rels_counts.items():
        if count > min_freq:
            rels_checklist.append((rel_token))

    return rels_checklist, relation_tokens

def filter_relations(rel_path, data_path, device, all_objects, rels_checklist):
    def crop_image(img,obj):
        return img[obj['y']:obj['y']+obj['h'],
                   obj['x']:obj['x']+obj['w'],:]
    print('extracting relations and images')

    resnet = models.resnet101(pretrained=True)
    resnet.to(device)
    resnet.eval()
    # relations_len = relations.shape[0]

    zfile = zipfile.ZipFile(rel_path)
    X, Y=[], []
    save_point = 10000
    save_point_idx = 0
    image_damage_cnt, object_id_absent_cnt, unknown_skip_cnt, relation_cnt = 0, 0, 0, 0
    all_objects_keys = set(all_objects.keys())
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        data = json.loads(ifile.read().decode('utf-8'))
        for idx, rels in tqdm(enumerate(data),total=len(data)):
            img_id = rels['image_id']
            image_path1 = VG_path+'VG_100K/{}.jpg'.format(img_id)
            image_path2 = VG_path+'VG_100K_2/{}.jpg'.format(img_id)
            image_path = image_path1 if os.path.isfile(image_path1) else image_path2
            img_pred = mpimg.imread(image_path)
            # filter if image is not damaged
            if len(img_pred.shape)<3:
                image_damage_cnt+=len(rels['relationships'])
                continue

            for rel in rels['relationships']:
                img_subj = crop_image(img_pred, rel['subject'])
                img_obj = crop_image(img_pred, rel['object'])

                if len(rel['predicate']) > 0:
                    # skip if there are more predicates for the relation
                    pred = clean_string(rel['predicate'])
                else:
                    continue

                if (0 in img_subj.shape) or (0 in img_obj.shape):
                    # skip if the image is not exist
                    continue

                # id could be not in all_objects keys
                # id could refer to empty list in all_objects
                subj_id = rel['subject']['object_id']
                obj_id = rel['object']['object_id']
                try:
                    for subj in all_objects[subj_id]:
                        for obj in all_objects[obj_id]:
                            if ((subj,pred) in rels_checklist) or ((pred, obj) in rels_checklist):
                                X.append([img_subj, img_pred, img_obj])
                                Y.append([subj.split('.')[0], pred, obj.split('.')[0]])
                                relation_cnt +=1
                except:
                    if (subj_id not in all_objects_keys) or (obj_id not in all_objects_keys):
                        object_id_absent_cnt +=1
                    else:
                        unknown_skip_cnt +=1
                        print(rel)
            
            if idx!=0 and (idx%save_point)==0:
                # run CNN feature extraction function
                # save parameters
                print('CNN extraction')
                # print(len(X))
                X = CNN_feature_extraction(X,resnet,device)
                print('saving data part {}'.format(save_point_idx))
                pickle.dump(X, open(data_path+"x_{}.p".format(save_point_idx), "wb"))
                pickle.dump(Y, open(data_path+"y_{}.p".format(save_point_idx), "wb"))
                X, Y=[],[]
                save_point_idx+=1

    print('unknown_failure_cnt: ', unknown_skip_cnt)
    print('key_absent_cnt: ', object_id_absent_cnt)
    print('include_relation_cnt: ', relation_cnt)
    


def CNN_feature_extraction(X, resnet, device):
    # tranforming step reqires large Mem, have to split the processing into steps to stabalized the mem required
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    image_batch_size = 10000
    output=[]

    for start_point in range(0,len(X),image_batch_size):
        end_point = start_point+image_batch_size if start_point+image_batch_size<len(X) else len(X) 
        X_transformed = []
        for img_triple in X[start_point:end_point]:
            X_transformed.append(torch.stack([transform(img) for img in img_triple]))  # append dimension 3*3*244*224
        X_transformed = torch.cat(X_transformed) # flatten the shape to (-1,3,224,224) to pass the CNN

        x_loader = DataLoader(X_transformed, batch_size = 1024)
        del X_transformed

        resnet.to(device)
        resnet.eval()
        with torch.no_grad():
            for x_batch in x_loader:
                output_batch = resnet(x_batch.to(device))
                output.append(output_batch)
        del x_loader

    return torch.cat(output).cpu().reshape(-1,3,1000)


def features_PCA(data_path, pca_path,pixie_dim_new):
    pixie_dim = 1000
    # cov_global = torch.zeros(pixie_dim,pixie_dim)
    data_size = 0
    batch_number = 10
    x_batch, y_batch = [], []
    for itr in tqdm(range(batch_number)):
        x_hidden = pickle.load(open(data_path+"x_{}.p".format(itr), "rb")).reshape(-1, pixie_dim)
        Y_flat = pickle.load(open(data_path+"y_{}.p".format(itr), "rb"))
        x_batch.append(x_hidden)
        y_batch.append(Y_flat)

    x_batch = torch.cat(x_batch)
    y_batch = np.concatenate(y_batch)

    # x_batch = (x_batch-torch.mean(x_batch,0))/torch.std(x_batch,0)
    cov = np.cov(x_batch.T)

    eigen_values, eigen_vectors = np.linalg.eig(cov)
    projection_matrix = (eigen_vectors.T[:][:pixie_dim_new]).T
    X_pca = x_batch.numpy().dot(projection_matrix) /pow(eigen_values[:pixie_dim_new],1/2)*1.13

    pickle.dump(torch.Tensor(X_pca).reshape(-1,3,pixie_dim_new), open(pca_path+"x_preprocessed.p", "wb"))
    pickle.dump(y_batch, open(pca_path+"y_preprocessed.p", "wb"))
            
def generate_vocab(data_path):
    # generate the vocabulary from the filtered pca transformed data.
    Y_flat = pickle.load(open(data_path+"y_preprocessed.p", "rb"))
    predicate_list, predicate_count = np.unique(np.array(Y_flat).reshape(-1),return_counts=True)
    predicates_table = {w:i for i,w in enumerate(predicate_list)}
    pickle.dump((predicate_list, predicates_table), open(data_path+'vocabulary.p','wb'))
    return predicate_list, predicates_table


def plot_example(idx, re_idx, work_path, relations):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixie_dim', type=int, default=100, help='dimension of pixie')
    # This is the directory of your objects.json and relationships.json files
    parser.add_argument('--VG_path', type=str, default='/local/scratch/yl535/visualgeno/', help='path to the VG data')
    # The directory of filtered and transformed data
    parser.add_argument('--data_path', type=str, default='pixie_data/', help='Path to save data')
    # The directory of the PCA transformed data
    parser.add_argument('--pca_path', type=str, default='data_pca/', help='Path to save PCA data')

    parser.add_argument('--pca_only', type=bool, default=True, help='If only runs the pca')
    parser.add_argument('--min_freq', type=int, default=30, help='Filtering out low frequent words')
    args = parser.parse_args()

    print('Processing the Visual Genome data')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not args.pca_only:
        if not os.path.isdir(args.data_path ): os.mkdir(args.data_path )
        obj_path = args.VG_path+'objects.json.zip'
        rel_path = args.VG_path+'relationships.json.zip'

        filtered_objects_counts, all_objects = generate_objects_vocab(obj_path, args.min_freq)
        rels_checklist, _ = generate_rels_checklist(rel_path, list(filtered_objects_counts.keys()), all_objects, args.min_freq)
        filter_relations(rel_path, args.data_path, device, all_objects, rels_checklist)

    if not os.path.isdir(args.pca_path): os.mkdir(args.pca_path)
    features_PCA(args.data_path, args.pca_path, pixie_dim_new=args.pixie_dim)

    _, _ = generate_vocab(args.pca_path)

