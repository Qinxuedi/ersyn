import pandas as pd
import numpy as np

hhh=pd.read_csv('csv_files/album_train_data.csv')
test_data=pd.read_csv('csv_files/album_test_data.csv')

m_data=pd.read_csv('csv_files/album_m_data.csv')
u_data=pd.read_csv('csv_files/album_u_data.csv')
m_len,u_len=len(m_data),len(u_data)
m_p=m_len/(m_len+u_len)
m_len,u_len,m_len+u_len,m_p,len(hhh)
m_array=np.array(m_data)[:,3:-1]
u_array=np.array(u_data)[:,3:-1]

m_data.shape, u_data.shape

test_array=np.array(test_data)[:,3:-1]
test_label=np.array(test_data)[:,-1]
test_label=test_label.astype(int)

print(test_label)
sum(test_label),len(test_label)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_array,u_array),axis=0)
y=np.array([1]*len(m_array)+[0]*len(u_array))
clf=RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('******************real**************')
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)

from sklearn.mixture import GaussianMixture
m_gm=GaussianMixture(n_components=10, random_state=0).fit(m_array)
print('m_gm.weights_:')
print(m_gm.weights_,'\n')
print('m_gm.means:')
print(m_gm.means_,'\n')
print('m_gm.covariances:')
print(m_gm.covariances_)

print(m_gm.score(m_array))
score=m_gm.score_samples(m_array)
sort_score_id=np.argsort(score)
for id in sort_score_id[:10]:
    print(id,m_array[id],score[id])
sort_m_array=np.concatenate((m_array[sort_score_id],np.array(m_data)[:,0:3],(score[sort_score_id])[:,np.newaxis]),axis=1)

from sklearn.mixture import GaussianMixture
u_gm=GaussianMixture(n_components=20, random_state=0).fit(u_array)
print('u_gm.weights_:')
print(u_gm.weights_,'\n')
print('u_gm.means:')
print(u_gm.means_,'\n')
print('u_gm.covariances:')
print(u_gm.covariances_)


print(u_gm.score(u_array))
score=u_gm.score_samples(u_array)
sort_score_id=np.argsort(score)
for id in sort_score_id[:10]:
    print(id,u_array[id],score[id])
sort_u_array=np.concatenate((u_array[sort_score_id],np.array(u_data)[:,0:3],(score[sort_score_id])[:,np.newaxis]),axis=1)
delta_u_array=u_array[sort_score_id[:50000]]
# print(delta_u_array)


A_NUM=6906
B_NUM=55922
MATCH_NUM=132
m_gm_sample_array, m_gm_sample_com=m_gm.sample(MATCH_NUM)
# u_gm_sample_array, u_gm_sample_com=u_gm.sample(B_NUM-MATCH_NUM)
u_gm_sample_array, u_gm_sample_com=u_gm.sample(u_data.shape[0])

print(u_gm.score(u_gm_sample_array))
score=u_gm.score_samples(u_gm_sample_array)
sort_score_id=np.argsort(score)
for id in sort_score_id[:10]:
    print(id,u_gm_sample_array[id],score[id])


m_clip_gm_sample_array=np.clip(m_gm_sample_array,0,1)
u_clip_gm_sample_array=np.clip(u_gm_sample_array,0,1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_clip_gm_sample_array,u_clip_gm_sample_array),axis=0)
y=np.array([1]*len(m_clip_gm_sample_array)+[0]*len(u_clip_gm_sample_array))
clf=RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('**************m_clip_gm_sample_array,u_clip_gm_sample_array**************')
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)

m_clip_gm_sample_array.shape, u_clip_gm_sample_array.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_clip_gm_sample_array,u_clip_gm_sample_array,delta_u_array),axis=0)
y=np.array([1]*len(m_clip_gm_sample_array)+[0]*len(u_clip_gm_sample_array)+[0]*len(delta_u_array))
clf=RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('**************m_clip_gm_sample_array,u_clip_gm_sample_array,delta_u_array**************')
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)


def find_repr_buckets(bucket_ids,buckets,repr_num):
    repr_buckets={}
    bucket_num=len(bucket_ids)
    bucket_ids_and_sizes=[]
    for bucket_id in bucket_ids:
        bucket_ids_and_sizes.append((bucket_id,len(buckets[bucket_id])))
    bucket_ids_and_sizes.sort(key=lambda x:x[1])
    sample_num=delta=0
    for i in range(bucket_num):
        if i==bucket_num-1:
            break
        bucket_id,bucket_size=bucket_ids_and_sizes[i]
        if delta==0:
            repr_buckets[bucket_id]=buckets[bucket_id][:]
            sample_num+=bucket_size
            delta=int((repr_num-sample_num)/(bucket_num-(i+1)))
            if delta<bucket_ids_and_sizes[i+1][1]:
                print('***************delta={}****************'.format(delta))
                continue
            else:
                delta=0
        else:
            repr_buckets[bucket_id]=np.random.choice(buckets[bucket_id],delta,replace=False)
            sample_num+=delta
        
    bucket_id,bucket_size=bucket_ids_and_sizes[bucket_num-1]     
    repr_buckets[bucket_id]=np.random.choice(buckets[bucket_id],repr_num-sample_num,replace=False)
    return repr_buckets
    
    
def representative_vector_by_bucket(vectors,repr_num):
    # get buckets
    buckets={}
    bucket_ids=[]
    length,dimension_num=vectors.shape
    for i in range(length):
        vector=vectors[i]
        bucket_id=[]
        for value in vector:
            bucket_id.append(int(value*10))
        bucket_id=tuple(bucket_id)
        if bucket_id in buckets:
            buckets[bucket_id].append(i)
        else:
            bucket_ids.append(bucket_id)
            buckets[bucket_id]=[i]
    bucket_ids.sort()
    
    # get representative_vector_by_bucket
    repr_buckets=find_repr_buckets(bucket_ids,buckets,repr_num)
    repr_vectors=np.empty((0,dimension_num))
    for bucket_id in bucket_ids:
        repr_vectors=np.concatenate((repr_vectors,vectors[repr_buckets[bucket_id]]))
    return bucket_ids,buckets,repr_buckets,repr_vectors
        
u_bucket_ids,u_buckets,u_repr_buckets,u_repr_vectors=representative_vector_by_bucket(u_clip_gm_sample_array,B_NUM-MATCH_NUM)
print(len(u_bucket_ids), u_repr_vectors.shape, B_NUM-MATCH_NUM)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_clip_gm_sample_array,u_repr_vectors),axis=0)
y=np.array([1]*len(m_clip_gm_sample_array)+[0]*len(u_repr_vectors))
clf=RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('**************m_clip_gm_sample_array,u_repr_vectors**************')
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)


uu_clip_gm_sample_array=np.concatenate((u_clip_gm_sample_array,delta_u_array))
uu_bucket_ids,uu_buckets,uu_repr_buckets,uu_repr_vectors=representative_vector_by_bucket(uu_clip_gm_sample_array,B_NUM-MATCH_NUM)
print(len(uu_bucket_ids), uu_repr_vectors.shape,B_NUM-MATCH_NUM)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
x=np.concatenate((m_clip_gm_sample_array,uu_repr_vectors),axis=0)
y=np.array([1]*len(m_clip_gm_sample_array)+[0]*len(uu_repr_vectors))
clf=RandomForestClassifier(max_depth=6, random_state=0)
clf.fit(x,y)
pred_label=clf.predict(test_array)
print('**************m_clip_gm_sample_array,uu_repr_vectors**************')
print('Accuracy', accuracy_score(test_label, pred_label))
print('Precision', precision_score(test_label, pred_label))
print('Recall', recall_score(test_label, pred_label))
print('f1_score', f1_score(test_label, pred_label))
confusion_matrix(test_label, pred_label)




album=pd.read_csv('csv_files/album.csv')

log=open('log.txt','w')
log.write('this is log file...\n')
log.close()

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


class DomainSamplePut(object):
    def __init__(self,csv_file):
        self.csv_file=csv_file
        self.columns=csv_file.columns
        self.size=len(self.csv_file)
        
    def sample(self):
        sample_index=np.random.randint(self.size)
        sample_object={}
        for column in self.columns:
            sample_object[column]=self.csv_file[column][sample_index]
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

import random
import math
import time
# import py_entitymatching as em


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 400):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src
    
    
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention
    
    
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
    
    

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 400):
        super().__init__()
        
        self.device = device
        self.output_dim=output_dim
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        
        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]
            
        return output, attention
    
    
class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
    
    
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    

def jac_qgm_3_qgm_3(seq1, seq2):
    qgm1 = []
    qgm2 = []
    seq1 = '##' + seq1 + '$$'
    seq2 = '##' + seq2 + '$$'
    for i in range(len(seq1) - 2):
        qgm1.append(seq1[i:i + 3])
    for i in range(len(seq2) - 2):
        qgm2.append(seq2[i:i + 3])
    qgm1 = set(qgm1)
    qgm2 = set(qgm2)
    return len(qgm1 & qgm2) / len(qgm1 | qgm2)


def cos_dlm_dc0_dlm_dc0(seq1, seq2):
    set1=set(seq1.split(' '))
    set2=set(seq2.split(' '))
    return float(len(set1 & set2)) / (math.sqrt(float(len(set1))) * math.sqrt(float(len(set2))))

def mel(seq1,seq2):
    return em.monge_elkan(seq1,seq2)

def lev_sim(seq1,seq2):
    return em.lev_sim(seq1,seq2)
    
def exm(seq1,seq2):
    return seq1 == seq2

def anm(seq1,seq2):
    return em.abs_norm(seq1,seq2)

    
def tokens_to_str(tokens):
    if tokens[-1]=='<eos>':
        return ''.join(tokens[:-1])
    return ''.join(tokens)

def tokenizer(text):
    return [char for char in text]


def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 400):
    
    model.eval()
        
    tokens = tokenizer(sentence)

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return tokens_to_str(trg_tokens[1:])


def translate_sentence_sample(sentence, src_field, trg_field, model, device, max_len = 400):
    
    model.eval()
        
    tokens = tokenizer(sentence)

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    # print('src_indexes:',src_indexes)
    # log=open('log.txt','a+')
    # log.write('src_indexes:{}\n'.format(src_indexes))
    # log.close()
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            # output = [1, trg len, output dim]
            # output[0,-1,:] = [output dim]
        
        prob = F.softmax(output[0,-1,:],dim=-1)
        # prob=[output dim]
        m=Categorical(prob)
        pred_token=m.sample()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return tokens_to_str(trg_tokens[1:])


    

import torch
import torch.nn as nn
import torch.optim as optim
import os
import torch.nn.functional as F
from torchtext.data import Field, Dataset, TabularDataset, BucketIterator
from torch.distributions import Categorical

import pandas as pd
import numpy as np

import random
import math
import time
# from model import Encoder, Decoder, Seq2Seq, translate_sentence, translate_sentence_sample

def tokenizer(text):
    return [char for char in text]


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


domain_to_seqs={}
domain_to_models={}

domain_to_loadEpoch={'song':{0:399,1:399,2:399,3:399,4:399,
                            5:399,6:399,7:399,8:399,9:399},
                    'album':{0:399,1:399,2:399,3:399,4:399,
                            5:399,6:399,7:399,8:399,9:399}}

for domain in ['song','album']:
    delta_to_seqs={}
    delta_to_models={}
    
    for delta_id in [0,1,2,3,4,5,6,7,8,9]:
        
        
        DELTA_ID=delta_id
        
        SEQ = Field(tokenize = tokenizer, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True,
                    batch_first = True)

        SIM = Field(sequential=False,
                use_vocab=False,
                dtype=torch.float32)
        
        transformer_data = TabularDataset('csv_files/seq2seq_{}_jac_delta_{}.csv'.format(domain,DELTA_ID),format='csv',skip_header=True,
                        fields=[('seq1',SEQ),('seq2',SEQ),('{}_{}_jac_qgm_3_qgm_3'.format(domain,domain),SIM)])

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
        LOAD_EPOCH = domain_to_loadEpoch[domain][DELTA_ID]
        
        
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
        
        print(domain,delta_id,INPUT_DIM,OUTPUT_DIM)
        
        transformer_model = Seq2Seq(enc, dec, SEQ_PAD_IDX, SEQ_PAD_IDX, device).to(device)
        transformer_model.load_state_dict(torch.load('models/{}/delta_{}_transformer_model_{}.pt'.format(domain,DELTA_ID, LOAD_EPOCH),map_location=device))
        delta_to_models[DELTA_ID]=transformer_model
        
    
    domain_to_seqs[domain]=delta_to_seqs
    domain_to_models[domain]=delta_to_models
    


unmatched_domain_to_seqs={}
unmatched_domain_to_models={}

unmatched_domain_to_loadEpoch={'song':{0:-1,1:322,2:399,3:399,4:399,
                            5:399,6:399,7:399,8:-1,9:-1},
                    'album':{0:399,1:399,2:399,3:399,4:399,
                            5:399,6:399,7:399,8:399,9:399}}

for domain in ['song','album']:
    unmatched_delta_to_seqs={}
    unmatched_delta_to_models={}
    
    for delta_id in [1,2,3,4,5,6,7]:
        
        
        DELTA_ID=delta_id
        
        SEQ = Field(tokenize = tokenizer, 
                    init_token = '<sos>', 
                    eos_token = '<eos>', 
                    lower = True,
                    batch_first = True)

        SIM = Field(sequential=False,
                use_vocab=False,
                dtype=torch.float32)
        
        transformer_data = TabularDataset('csv_files/seq2seq_{}_jac_delta_{}_unmatched.csv'.format(domain,DELTA_ID),format='csv',skip_header=True,
                        fields=[('seq1',SEQ),('seq2',SEQ),('{}_{}_jac_qgm_3_qgm_3'.format(domain,domain),SIM)])

        SEQ.build_vocab(transformer_data.seq1, transformer_data.seq2)
        unmatched_delta_to_seqs[DELTA_ID]=SEQ
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
        LOAD_EPOCH = unmatched_domain_to_loadEpoch[domain][DELTA_ID]
        
        
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
        
        print(domain,delta_id,INPUT_DIM,OUTPUT_DIM)
        
        transformer_model = Seq2Seq(enc, dec, SEQ_PAD_IDX, SEQ_PAD_IDX, device).to(device)
        transformer_model.load_state_dict(torch.load('models/{}/delta_{}_transformer_unmatched_model_{}.pt'.format(domain,DELTA_ID, LOAD_EPOCH),map_location=device))
        unmatched_delta_to_models[DELTA_ID]=transformer_model
        
    
    unmatched_domain_to_seqs[domain]=unmatched_delta_to_seqs
    unmatched_domain_to_models[domain]=unmatched_delta_to_models
    


def generate_sim_sentence(domain,sen_a,sim,sample_time,device):
    DELTA_ID=int(sim*10)  
    if sim<=0:
        DELTA_ID=0
    if sim>=1:
        DELTA_ID=9
    SEQ=domain_to_seqs[domain][DELTA_ID]
    transformer_model=domain_to_models[domain][DELTA_ID]
    best_sen_b=''
    best_sim_sen=10
    for i in range(sample_time):
        # print('!',i)
        # log=open('log.txt','a+')
        # log.write('! {}\n'.format(i))
        # log.close()
        now_sen_b=translate_sentence_sample(sen_a,SEQ,SEQ,transformer_model, device)
        # print(now_sen_b)
        # log=open('log.txt','a+')
        # log.write('{}\n'.format(now_sen_b))
        # log.close()
        now_sim_sen=jac_qgm_3_qgm_3(sen_a,now_sen_b)
        if abs(now_sim_sen-sim)<abs(best_sim_sen-sim):
            best_sen_b=now_sen_b
            best_sim_sen=now_sim_sen
    return best_sen_b


def generate_sim_unmatched_sentence(domain,sen_a,sim,sample_time,device):
    DELTA_ID=int(sim*10)  
    if sim<=0:
        DELTA_ID=0
    if sim>=1:
        DELTA_ID=9
    SEQ=unmatched_domain_to_seqs[domain][DELTA_ID]
    transformer_model=unmatched_domain_to_models[domain][DELTA_ID]
    best_sen_b=''
    best_sim_sen=10
    for i in range(sample_time):
        # print('#',i)
        # log=open('log.txt','a+')
        # log.write('# {}\n'.format(i))
        # log.close()
        now_sen_b=translate_sentence_sample(sen_a,SEQ,SEQ,transformer_model, device)
        # print(now_sen_b)
        # log=open('log.txt','a+')
        # log.write('{}\n'.format(now_sen_b))
        # log.close()
        now_sim_sen=jac_qgm_3_qgm_3(sen_a,now_sen_b)
        if abs(now_sim_sen-sim)<abs(best_sim_sen-sim):
            best_sen_b=now_sen_b
            best_sim_sen=now_sim_sen
    return best_sen_b



# from model import translate_sentence, translate_sentence_sample
# from model import jac_qgm_3_qgm_3
import random


min_A_price,max_A_price=0.99,1.29
min_B_price,max_B_price=0.99,1.29

DS=DomainSample(album)

gen_A=open('csv_files/gen_album_A_.csv','w')
gen_B=open('csv_files/gen_album_B_.csv','w')
gen_matching=open('csv_files/gen_album_matching_.csv','w')

gen_A.write('"id","Song_Name","Artist_Name","Album_Name","Price"\n')
gen_B.write('"id","Song_Name","Artist_Name","Album_Name","Price"\n')


gen_A.close()
gen_B.close()
gen_matching.close()

id_a=id_b=0
sample_time=10

MAX_LENGTH=350



print('**************Step 1**************')
for i in range(MATCH_NUM):
    print(i,end=' ')
    sim_vec=m_clip_gm_sample_array[i]
    sim_song,sim_artist,sim_album=sim_vec[0],sim_vec[1],sim_vec[2]
    
    sample_object=DS.sample() 

    song_a=sample_object['song']
    song_b=generate_sim_sentence('song',song_a,sim_song,sample_time,device)

    artist_a=sample_object['artist']
    artist_b=generate_sim_sentence('artist',artist_a,sim_artist,sample_time,device)

    album_a=sample_object['album']
    album_b=generate_sim_sentence('album',album_a,sim_album,sample_time,device)

    price_a=price_b=random.uniform(min_A_price,max_A_price)

    
    gen_A=open('csv_files/gen_album_A_.csv','a+')
    gen_A.write('{},"{}","{}","{}",{}\n'.format(str(id_a),song_a,artist_a,album_a,price_a))
    gen_A.close()
    id_a+=1

    gen_B=open('csv_files/gen_album_B_.csv','a+')
    gen_B.write('{},"{}","{}","{}",{}\n'.format(str(id_b),song_b,artist_b,album_b,price_b))
    gen_B.close()
    id_b+=1



