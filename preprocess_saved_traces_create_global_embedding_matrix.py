

import glob


#this version is for 1 trace/file processing only!!!

main_dir='/global/cscratch1/sd/leishao/etalumis_data_july30'#_small' #to test code's correctness
source_trace_dir='/trace_cache_orig'
target_trace_dir='/trace_cache'
file_list=glob.glob(main_dir+ source_trace_dir + '/pyprob_traces_1_*')
#file_list=glob.glob(main_dir+ trace_dir + '/pyprob_traces_100_*')



print("number of files=",len(file_list))



import pyprob
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus']=False
from matplotlib.ticker import MultipleLocator
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#get_ipython().magic('matplotlib inline')

min_energy_deposit = 0.05  # caloutils::minEnergyDeposit in sharpa_tau_decay.cpp



import tarfile
import tempfile
import shutil
import uuid
import os
from collections import OrderedDict
from pyprob.nn import SampleEmbeddingFC, ProposalCategorical, ProposalNormal, ProposalUniform, ProposalUniformMixture, ProposalUniformKumaraswamyMixture, ProposalUniformKumaraswamy, ProposalPoisson
from pyprob.distributions import Categorical, Mixture, Normal, TruncatedNormal, Uniform, Poisson, Kumaraswamy
from pyprob import util
from torch.nn import Parameter


traces_max_length=0
traces_lengths=[]
sb={}
addr_to_ix={}
address_stats=OrderedDict()
distribution_type_embeddings={}
sample_embedding_layers={} #map address to sample embedding layer in dictionary
proposal_layers={} #map address to proposal layer in dictionary
address_embeddings={}
sample_embedding_dim=16
lstm_dim=256
distribution_type_embedding_dim=4
address_embedding_dim=128



import sys
def group_traces(traces):
    global traces_max_length
    global addr_to_ix
    global address_stats
    for trace in traces: 
        if trace.length == 0:
            print("find one trace with length 0")
            raise ValueError('Trace of length zero')
        traces_lengths.append(trace.length)
        if trace.length > traces_max_length:
            traces_max_length = trace.length
            #print("new max trace length is ", traces_max_length)
        for idx, sample in enumerate(trace.samples): #only use controlled samples to build the address embedding matrix
            address = sample.address
            distribution=sample.distribution
            distribution_type=sample.distribution.name
            if address not in address_stats:
                address_id=str(len(address_stats))#not add A, not +1 for matrix
                addr_to_ix[address]=address_id
                t=Parameter(util.Tensor(address_embedding_dim).normal_())
                address_embeddings[address]=t
                address_stats[address]=[1, address_id, sample.distribution.name, sample.control, sample.replace, sample.observed]
                #print('enter address_stats, idx=', idx, ', address_id=', address_id, ',control=', sample.control, ',replace=', sample.replace, ',observed=', sample.observed, ',distribution=', sample.distribution.name, ',distribution_length_categories=', distribution.length_categories if isinstance(distribution, Categorical) else -1)
               
                if distribution_type not in distribution_type_embeddings:
                    #print('New distribution type: {}'.format(distribution_type))
                    i = len(distribution_type_embeddings)
                    if i < distribution_type_embedding_dim:
                        t = util.one_hot(distribution_type_embedding_dim, i)
                        distribution_type_embeddings[distribution_type] = t
                            #print('sie of distribution type one hot embedding={}'.format(t.size()))
                    else:
                        print('Warning: overflow (collision) in distribution type embeddings. Allowed: {}; Encountered: {}'.format(distribution_type_embedding_dim, i + 1))
                        distribution_type_embeddings[distribution_type] = random.choice(list(distribution_type_embeddings.values()))

                if isinstance(distribution, Categorical):
                    sample_embedding_layer=SampleEmbeddingFC(sample.value.nelement(),sample_embedding_dim, input_is_one_hot_index=True, input_one_hot_dim=sample.distribution.length_categories)
                else:
                    sample_embedding_layer=SampleEmbeddingFC(sample.value.nelement(), sample_embedding_dim)
                sample_embedding_layers[address]=sample_embedding_layer
                if isinstance(distribution, Categorical):
                    proposal_layer = ProposalCategorical(lstm_dim, distribution.length_categories)
                elif isinstance(distribution, Normal):
                    proposal_layer = ProposalNormal(lstm_dim, distribution.length_variates)
                elif isinstance(distribution, Uniform):
                    proposal_layer = ProposalUniformKumaraswamyMixture(lstm_dim)
                elif isinstance(distribution, Poisson):
                    proposal_layer = ProposalPoisson(lstm_dim)
                else:
                    raise ValueError('Unsupported distribution: {}'.format(distribution.name))
                proposal_layers[address]=proposal_layer

            else:
                address_stats[address][0]+=1
                address_id=str(list(address_stats.keys()).index(address))
                #print('meet existing one at index=', idx, 'address_id=', address_id,',observed=', sample.observed, ',distribution=', sample.distribution.name, ', distribtution_length_categorical=', distribution.length_categories if isinstance(distribution, Categorical) else -1)
                
        h = trace.length#hash(trace.addresses())
        if h not in sb:
            sb[h] = []
        sb[h].append(trace) 

def load_traces_save_to_dir(file_name, index):
    tar = tarfile.open(file_name, 'r:gz')
    tmp_dir = tempfile.mkdtemp(suffix=str(uuid.uuid4()))
    tmp_file = os.path.join(tmp_dir, 'pyprob_traces')
    tar.extract('pyprob_traces', tmp_dir)
    tar.close()
    data = torch.load(tmp_file, map_location=lambda storage, loc: storage)
    shutil.rmtree(tmp_dir)
    traces = data['traces']
    #below for 1 trace for file only!!!
    len_trace=traces[0].length
    path=main_dir+ target_trace_dir + '/tracelen_' + str(len_trace)
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + '/trace'+str(index)+ '_len' + str(len_trace) +'.pt'
    torch.save(traces[0], filename)
  
        
    return traces


traces_max_length=0
traces_lengths=[]
sb={}
count_traces=0
address_stats=OrderedDict()
for i, file_name in enumerate(file_list):
    #print('file index=', i)
    traces=load_traces_save_to_dir(file_name, i)
    count_traces+=len(traces)
    group_traces(traces)



print("total traces in this dataset=", count_traces)
print("total addresses for controlled samples in this dataset=", len(address_stats))



print("max length of the traces = ", traces_max_length)



print("address embedding matrix's vocabulary size =", len(addr_to_ix))



traces_lengths=np.array(traces_lengths)
plt.hist(traces_lengths,normed=False, bins=100)
plt.xlabel('trace_length')
plt.ylabel('count')
plt.title('Histogram ')



plt.savefig(main_dir + '/trace_length_histogram.png')



torch.save(address_embeddings, main_dir+'/address_embeddings.pt')



print("address embedding matrix's vocabulary size =", len(addr_to_ix))



torch.save(sample_embedding_layers, main_dir+'/sampleEmbeddinglayers.pt')



torch.save(proposal_layers, main_dir+'/proposal_layers.pt')



torch.save(distribution_type_embeddings, main_dir+'/distribution_type_embeddings.pt')



torch.save(addr_to_ix, main_dir+'/addr_to_ix.pt')



torch.save(address_stats, main_dir+'/address_stats.pt')



import torch.nn as nn
from torch.nn import Parameter
address_embedding_dim=128
all_addr_num=len(addr_to_ix)
addr_embeddings = nn.Embedding(all_addr_num, address_embedding_dim)
t=Parameter(util.Tensor((all_addr_num, address_embedding_dim)).normal_())
addr_embeddings=t



torch.save(addr_embeddings, main_dir+'/addr_embeddings_idx.pt')



sorted_traces_lengths=sorted(traces_lengths, reverse=False)



my_unique_tracelen_list=sorted(set(traces_lengths))

import math
num_buckets=10
num_trace_per_bucket=math.floor(len(file_list)/num_buckets)



torch.save(my_unique_tracelen_list, main_dir + 'my_unique_tracelen_list.pt')



torch.save(sorted_traces_lengths, main_dir + 'sorted_traces_lengths.pt')



def trace_folder_current_files(dirname, discarded_file_names):
    files = [name for name in os.listdir(dirname)]
    files = list(map(lambda f: os.path.join(dirname, f), files))
    for discarded_file_name in discarded_file_names:
        if discarded_file_name in files:
            files.remove(discarded_file_name)
    return files



from shutil import copyfile
unique_sorted_list_idx=0
bucket_idx=0
temp_buffer=[]
discarded_file_names=[]
discard_source = True
while (bucket_idx < num_buckets):
    print('bucket_idx={}'.format(bucket_idx))
    print('unique_sorted_list_idx={}'.format(unique_sorted_list_idx))
    while (len(temp_buffer) < num_trace_per_bucket):
        dirname= main_dir + target_trace_dir + '/tracelen_' + str(my_unique_tracelen_list[unique_sorted_list_idx])
        current_files= trace_folder_current_files(dirname, discarded_file_names)
        unique_sorted_list_idx +=1
        temp_buffer.extend(current_files)
    used_files=temp_buffer[0:num_trace_per_bucket]
    #print(used_files)
    bucket_dir=main_dir + target_trace_dir +'/bucket_' + str(bucket_idx)
    if not os.path.exists(bucket_dir):
        os.makedirs(bucket_dir)
    for full_filename in used_files:
        copyfile(full_filename, bucket_dir + '/' + os.path.basename(full_filename))
    bucket_idx+=1
    if discard_source:
        discarded_file_names.extend(used_files)
    temp_buffer[0:num_trace_per_bucket]=[]
    






