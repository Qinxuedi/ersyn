import pandas as pd
import numpy as np

hhh=pd.read_csv('csv_files/hhh_blocking_restaurant.csv')
original_hhh=hhh.copy(deep=True)

def split_train_test(df,test_len):
    df_index=df.index
    df_len=len(df_index)
    test_index=np.random.choice(df_index,test_len,replace=False)
    test_index=np.sort(test_index)
    train_index=[]
    df_ptr=test_ptr=0
    while df_ptr<df_len and test_ptr<test_len:
        if df_index[df_ptr]==test_index[test_ptr]:
            df_ptr+=1
            test_ptr+=1
        elif df_index[df_ptr]<test_index[test_ptr]:
            train_index.append(df_index[df_ptr])
            df_ptr+=1
        else:
            test_ptr+=1
    while df_ptr<df_len:
        train_index.append(df_index[df_ptr])
        df_ptr+=1
    train_index=np.array(train_index)
    return test_index, train_index
 
test_percentage=0.3
test_num=int(len(hhh)*test_percentage)
test_index,train_index = split_train_test(hhh,test_num)

hhh=hhh.loc[train_index]

m_data=hhh[hhh.gold==1]
u_data=hhh[hhh.gold==0]
m_len,u_len=len(m_data),len(u_data)
m_p=m_len/(m_len+u_len)
m_len,u_len,m_len+u_len,m_p,len(hhh)

def calculate_mean_cov(array):
    m,n=array.shape  # m: number of tuples     n: number of dimensions
    mean=np.sum(array,axis=0)/m
    cov=np.zeros((n,n))
    for x in array:
        tmp=x-mean
        cov=cov+tmp.reshape(n,1)*tmp.reshape(1,n)
    cov=cov/m
    return mean, cov

m_array=np.array(m_data)[:,3:-1]
u_array=np.array(u_data)[:,3:-1]
m_mean,m_cov=calculate_mean_cov(m_array)
u_mean,u_cov=calculate_mean_cov(u_array)


def sample_from_distribution(m_mean,m_cov,u_mean,u_cov,m_sample_num,u_sample_num):
    m_sample_array=np.random.multivariate_normal(m_mean,m_cov,m_sample_num)
    u_sample_array=np.random.multivariate_normal(u_mean,u_cov,u_sample_num)
    return m_sample_array, u_sample_array

m_mean=m_mean.astype('float64')
m_cov=m_cov.astype('float64')
u_mean=u_mean.astype('float64')
u_cov=u_cov.astype('float64')
m_sample_num=m_len
u_sample_num=u_len
m_sample_array, u_sample_array=sample_from_distribution(m_mean,m_cov,u_mean,u_cov,m_sample_num,u_sample_num)

m_sample_mean,m_sample_cov=calculate_mean_cov(m_sample_array)
u_sample_mean,u_sample_cov=calculate_mean_cov(u_sample_array)

print(m_sample_mean,'\n')
print(m_sample_cov,'\n')
print(u_sample_mean,'\n')
print(u_sample_cov)
print(m_sample_array.shape)
print(u_sample_array.shape)

print(m_mean,'\n')
print(m_cov,'\n')
print(u_mean,'\n')
print(u_cov)
print(m_array.shape)
print(u_array.shape)

print(test_index)
test_data=original_hhh.loc[test_index]
test_array=np.array(test_data)[:,3:-1]
test_label=np.array(test_data)[:,-1]
test_label=test_label.astype(int)

print(test_label)
sum(test_label),len(test_label)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_array,u_array),axis=0)
y=np.array([1]*len(m_array)+[0]*len(u_array))
clf=RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)

from scipy.stats import multivariate_normal
pdf=multivariate_normal.pdf(u_array[:,:-1], mean=u_mean[:-1], cov=u_cov[:-1,:-1])    
sort_pdf_id=np.argsort(pdf)
sort_pdf=pdf[sort_pdf_id]
sort_pdf.resize(pdf.shape[0],1)
sort_u_array=np.concatenate((u_array[sort_pdf_id],sort_pdf),axis=1)


pdf=multivariate_normal.pdf(m_array[:,:-1], mean=m_mean[:-1], cov=m_cov[:-1,:-1])
sort_pdf_id=np.argsort(pdf)
sort_pdf=pdf[sort_pdf_id]
sort_pdf.resize(pdf.shape[0],1)
sort_m_array=np.concatenate((m_array[sort_pdf_id],sort_pdf),axis=1)

x_sample=np.concatenate((m_sample_array,u_sample_array),axis=0)
y_sample=np.array([1]*len(m_sample_array)+[0]*len(u_sample_array))
clf=RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_sample, y_sample)
pred_label=clf.predict(test_array)
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)

from scipy.stats import multivariate_normal
pdf=multivariate_normal.pdf(u_sample_array[:,:-1], mean=u_sample_mean[:-1], cov=u_sample_cov[:-1,:-1])
sort_pdf_id=np.argsort(pdf)
sort_pdf=pdf[sort_pdf_id]
sort_pdf.resize(pdf.shape[0],1)
sort_u_sample_array=np.concatenate((u_sample_array[sort_pdf_id],sort_pdf),axis=1)

pdf=multivariate_normal.pdf(m_sample_array[:,:-1], mean=m_sample_mean[:-1], cov=m_sample_cov[:-1,:-1])
sort_pdf_id=np.argsort(pdf)
sort_pdf=pdf[sort_pdf_id]
sort_pdf.resize(pdf.shape[0],1)
sort_m_sample_array=np.concatenate((m_sample_array[sort_pdf_id],sort_pdf),axis=1)

zomato=pd.read_csv('csv_files/zomato.csv')

class DomainSample(object):
    def __init__(self, csv_file):
        self.csv_file=csv_file
        self.columns=csv_file.columns
        self.index_pool=list(csv_file.index)
        self.index_pool_size=len(self.index_pool)
        
    def sample(self):
        sample_index=np.random.choice(self.index_pool_size,1)[0]
        sample_object={}
        for column in self.columns:
            sample_object[column]=self.csv_file[column][self.index_pool[sample_index]]
        del self.index_pool[sample_index]
        self.index_pool_size-=1
        return sample_object

class CategoricalDomainGenerate(object):
    def __init__(self, sim_to_value):
        self.sim_to_value=sim_to_value
        
    def generate(self,exp_sim):
        sim_delta=[]
        for sim in self.sim_to_value.keys():
            sim_delta.append((sim,abs(exp_sim-sim)))
        sim_delta.sort(key=lambda x:x[1])
        near_exp_sim=sim_delta[0][0]
        candidate_len=len(self.sim_to_value[near_exp_sim])
        sample_id=random.randint(0,candidate_len-1)
        return self.sim_to_value[near_exp_sim][sample_id]

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, Dataset, TabularDataset, BucketIterator
from torch.distributions import Categorical

import pandas as pd
import numpy as np
import spacy

import random
import math
import time
import dill
from model import Encoder, Decoder, Seq2Seq, translate_sentence, translate_sentence_sample

def tokenizer(text):
    return [char for char in text]


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

domain_to_seqs={}
domain_to_models={}

for domain in ['name','address']:
    delta_to_seqs={}
    delta_to_models={}

    for delta_id in [3,4,5,6,7,8,9]:

        DELTA_ID=delta_id

        SEQ = Field(tokenize = tokenizer, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True,
                    batch_first = True)

        SIM = Field(sequential=False,
                use_vocab=False,
                dtype=torch.float32)

        transformer_data = TabularDataset('csv_files/seq2seq_authors_jac_delta_{}.csv'.format(DELTA_ID),format='csv',skip_header=True,
                        fields=[('seq1',SEQ),('seq2',SEQ),('authors_authors_jac_qgm_3_qgm_3',SIM)])

        SEQ.build_vocab(transformer_data.seq1, transformer_data.seq2)
        delta_to_seqs[DELTA_ID]=SEQ
        SEQ_PAD_IDX = SEQ.vocab.stoi['<pad>']

        INPUT_DIM = len(SEQ.vocab)
        OUTPUT_DIM = len(SEQ.vocab)
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        LOAD_EPOCH = 399


        enc = Encoder(INPUT_DIM, 
                  HID_DIM, 
                  ENC_LAYERS, 
                  ENC_HEADS, 
                  ENC_PF_DIM, 
                  ENC_DROPOUT, 
                  device)

        dec = Decoder(OUTPUT_DIM, 
                      HID_DIM,
                      DEC_LAYERS, 
                      DEC_HEADS, 
                      DEC_PF_DIM, 
                      DEC_DROPOUT, 
                      device)

        transformer_model = Seq2Seq(enc, dec, SEQ_PAD_IDX, SEQ_PAD_IDX, device).to(device)
        transformer_model.load_state_dict(torch.load('models/name/delta_{}_transformer_model_{}.pt'.format(DELTA_ID, LOAD_EPOCH),map_location=device))
        delta_to_models[DELTA_ID]=transformer_model
        
    domain_to_seqs[domain]=delta_to_seqs
    domain_to_models[domain]=delta_to_models


from model import translate_sentence, translate_sentence_sample
from model import jac_qgm_3_qgm_3

DELTA_ID=4
SEQ=domain_to_seqs['name'][DELTA_ID]
transformer_model=domain_to_models['name'][DELTA_ID]
s1='interview with jim gray'
s2=translate_sentence_sample(s1, SEQ, SEQ, transformer_model, device)
s2,jac_qgm_3_qgm_3(s1,s2)

def generate_sim_sentence(domain,sen_a,sim,sample_time,device):
    DELTA_ID=int(sim*10)   
    SEQ=domain_to_seqs[domain][DELTA_ID]
    transformer_model=domain_to_models[domain][DELTA_ID]
    best_sen_b=''
    best_sim_sen=10
    for i in range(sample_time):
        now_sen_b=translate_sentence_sample(sen_a,SEQ,SEQ,transformer_modelmer_model, device)
        now_sim_sen=jac_qgm_3_qgm_3(sen_a,now_sen_b)
        if abs(now_sim_sen-sim)<abs(best_sim_sen-sim):
            best_sen_b=now_sen_b
            best_sim_sen=now_sim_sen
    return best_sen_b

from model import translate_sentence, translate_sentence_sample
from model import jac_qgm_3_qgm_3

DS=DomainSample(golden_records_cleaned)

city_sim_to_value={}
city_CDG=CategoricalDomainGenerate(city_sim_to_value)
flavor_sim_to_value={}
flavor_CDG=CategoricalDomainGenerate(flavor_sim_to_value)

gen=open('csv_files/gen.csv','w')
gen_matching=open('csv_files/gen_matching.csv','w')

gen.write('"id","name","address","city","flavor","cluster_id"\n')
gen_matching.write('"id_A","id_B","expected_sim_name","real_sim_name","expected_sim_address","real_sim_address","expected_sim_city","real_sim_city","expected_sim_flavor","real_sim_flavor"\n')

id=0
cluster_id=0
sample_time=10
for i in range(m_sample_num):
    print(i,end=' ')
    sim_vec=m_sample_array[i]
    sim_name,sim_address,sim_city,sim_flavor=sim_vec[0],sim_vec[1],sim_vec[2],sim_vec[3]
    
    sample_object=DS.sample()
    name_a=sample_object['name']
    name_b=generate_sim_sentence('name',name_a,sim_name,sample_time,device)
    address_a=sample_objectct['address']
    address_b=generate_sim_sentence('address',address_b,sim_address,sample_time,device)
    city_a,city_b=city_CDG.generate(sim_city)
    flavor_a,flavor_b=flavor_CDG.generate(sim_flavor)
    
    gen_matching.write('"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(
        str(id),str(id+1),sim_name,jac_qgm_3_qgm_3(name_a, name_b),sim_address,jac_qgm_3_qgm_3(address_a,address_b),
    sim_city,jac_qgm_3_qgm_3(city_a,city_b),sim_flavor,jac_qgm_3_qgm_3(flavor_a,flavor_b)))
    
    gen.write('{},"{}","{}","{}","{}",{}\n'.format(id,name_a,address_a,city_a,flavor_a,cluster_id))
    id+=1
    gen.write('{},"{}","{}","{}","{}",{}\n'.format(id,name_a,address_a,city_a,flavor_a,cluster_id))
    id+=1
    cluster_id+=1
    
gen.close()
    

