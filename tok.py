# https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41

import numpy as np

from numpy import dot
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel

pubmed_bert_model = AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
pubmed_bert_tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

# kg='[S] virus [R] co-occurs_with [O] infectious [R] associated_with [O] strain [R] co-occurs_with [O] minor [R] co-occurs_with [O] acute disease[S] identity [R] co-occurs_with [O] strain [R] associated_with [O] acute disease [R] associated_with [O] reported [R] co-occurs_with [O] bronchitis [R] associated_with [O] other [R] associated_with [O] field'

kg='[S] virus [R] co-occurs_with [O] infectious [R] associated_with [O] strain [R] co-occurs_with'
text="hello doctor how are you"

print("Before addind special tokens")
print(pubmed_bert_tokenizer.tokenize(kg))

print("Before adding pad and len")
print(pubmed_bert_tokenizer.tokenize(kg, max_length=40, padding='max_length', truncation=True))




# # new_tokens = ['uses','treats','prevents','isa','diagnoses','co-occurs_with','associated_with','affects']
# new_tokens = ['_NAF_R','_NAF_H', '_NAF_O','[S]','[R]','[O]']


# print("[ BEFORE ] tokenizer vocab size:", len(pubmed_bert_tokenizer)) 
# added_tokens = pubmed_bert_tokenizer.add_tokens(new_tokens)

# print("[ AFTER ] tokenizer vocab size:", len(pubmed_bert_tokenizer)) 
# print()
# print('added_tokens:',added_tokens)
# print()

# # resize the embeddings matrix of the model 
# pubmed_bert_model.resize_token_embeddings(len(pubmed_bert_tokenizer))


# # Let's call tokenizer_exBERT our tokenizer with the 2 new tokens.

# tokenizer_exBERT = pubmed_bert_tokenizer

# # tokenization of the text
# tokens = tokenizer_exBERT.tokenize(kg)
# print("Before addind special tokens")
# print(tokens)

