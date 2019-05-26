from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import logging
import pickle


MAX_SEQUENCE_LENGTH = 50
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.05




# read word embeddings from glove
def build_index_from_glove(glove_path: str):
    logging.info('Reading GloVe file from {}'.format(glove_path))
    word_to_embedding = {}
    with open(glove_path) as f:
        for line in f:
            word, embedding = line.split(maxsplit=1)
            embedding = np.fromstring(embedding, 'f', sep=' ')
            word_to_embedding[word] = embedding
    logging.info('Success! Retrieved {} words and their embeddings'.format(word_to_embedding.__len__()))
    return word_to_embedding




def read_train_set(data_path: str):
    picklef = open(data_path, 'rb')
    pf_dict = pickle.load(picklef)
    return pf_dict

def preprocess_train_data(data_dict):
    claim_data = []
    evidence_data = []
    labels = []
    e_len = []
    for line in data_dict:
        edata = []
        elabel = []
        counter = 0
        for e in line['evidence']:
            if counter >= 4 and e[1] == 0:
                counter+=1
                continue
            e_len.append(len(e[0]))
            edata.append(e[0])
            elabel.append(e[1])
            counter += 1

        cdata = [line['claim']]*len(edata)

        claim_data.extend(cdata)
        evidence_data.extend(edata)
        labels.extend(elabel)
    all_texts = claim_data + evidence_data
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(all_texts)
    del all_texts
    claim_sequences = tokenizer.texts_to_sequences(claim_data)
    evidence_sequences = tokenizer.texts_to_sequences(evidence_data)
    print(tokenizer.word_index.__len__())
    print(evidence_sequences.__len__())
    labels = to_categorical(np.asarray(labels))
    return claim_sequences, evidence_sequences, labels, tokenizer.word_index



def pad_data(seqs, maxlen):
    data = pad_sequences(seqs, maxlen=maxlen)
    return data


def split_train_vali(data_len, vali_percent):
    vali_split_num = int(vali_percent*data_len)
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    train_indices = indices[:-vali_split_num]
    vali_indices = indices[-vali_split_num:]

    return train_indices, vali_indices



def build_embedding_matrix(word_index, word_to_embedding):
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = word_to_embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def preprocess_train_data_for_verifyclaim(data_dict):
    claim_data = []
    evidence_data = []
    labels = []
    e_len = []
    for line in data_dict:
        edata = []
        elabel = []
        linelabel = line['label']
        if linelabel == 'NO ENOUGH INFO':
            continue
        if linelabel == 'SUPPORTS':
            linelabel = 1
        elif linelabel == 'REFUTES':
            linelabel = 0
        else:
            continue

        for e in line['evidence']:
            if e[1] == 1:
                e_len.append(len(e[0]))
                edata.append(e[0])
                elabel.append(linelabel)
        cdata = [line['claim']]*len(edata)

        claim_data.extend(cdata)
        evidence_data.extend(edata)
        labels.extend(elabel)
    all_texts = claim_data + evidence_data
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(all_texts)
    del all_texts
    claim_sequences = tokenizer.texts_to_sequences(claim_data)
    evidence_sequences = tokenizer.texts_to_sequences(evidence_data)
    print(tokenizer.word_index.__len__())
    print(evidence_sequences.__len__())
    labels = to_categorical(np.asarray(labels))
    return claim_sequences, evidence_sequences, labels, tokenizer.word_index

