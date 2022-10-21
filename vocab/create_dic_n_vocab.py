import json
import re
import random
import pickle
import time 

from transformers import AutoTokenizer, AutoModel




### load berttokenizer and save vocab 
# pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer.save_vocabulary('./pubmed_bert/')


relation_list = []
relation_list = ['uses','treats','prevents','isa','diagnoses','co-occurs_with','associated_with','affects']
relation_list += ['_NAF_R','_NAF_H', '_NAF_O','[S]','[R]','[O]']


print('rr len: ',len(relation_list))


print("pubmed_bert vocab load\n")
f_v = open('pubmed_bert/vocab.txt')
vocab = f_v.read().split('\n')
print(vocab[-10:])



new_vocab = vocab[:-1] + list(new_entity_list)
print("new vocab\n")
print(new_vocab[-10:])
print("new vocab len: ",len(new_vocab))

new_vocab = list(set(new_vocab))
print("new vocab len: ",len(new_vocab))

f_voc = open('pubmed_bert/new_vocab.txt','w')

for i,item in enumerate(new_vocab):
    # print("new_vocab:", item)
    f_voc.write(str(item))
    f_voc.write('\n')
f_voc.close()
