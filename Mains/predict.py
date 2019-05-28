from __future__ import print_function

from sys import argv
import numpy as np
import tensorflow as tf

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from Utils.preprocessing import *
from Models.EvidenceScoring import *
import json
from keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def min_index_replace(data):
    new_data = []
    for x in data:
        new_x = []
        for y in x:
            if y <= 20000:
                new_x.append(y)
            else:
                new_x.append(0)
        new_data.append(new_x)
    return new_data

def general_predict(model, tokenizer, claim, evidence, getArgmax=True):

    claim_seq = min_index_replace(tokenizer.texts_to_sequences(claim))
    evidence_seq = min_index_replace(tokenizer.texts_to_sequences(evidence))
    claim_input = pad_data(claim_seq, maxlen=50)
    evidence_input = pad_data(evidence_seq, maxlen=50)

    preds = model.predict([claim_input, evidence_input])
    if getArgmax:
        preds = np.argmax(preds, axis=1)
    else:
        return preds
    return preds


def tri_general_predict(model, tokenizer, claim, evidence, getArgmax=True):

    claim_seq = min_index_replace(tokenizer.texts_to_sequences(claim))
    evidence_seq = min_index_replace(tokenizer.texts_to_sequences(evidence))
    claim_input = pad_data(claim_seq, maxlen=100)
    evidence_input = pad_data(evidence_seq, maxlen=100)

    preds = model.predict([claim_input, evidence_input])
    if getArgmax:
        preds = np.argmax(preds, axis=1)
    else:
        return preds
    return preds

# if __name__ == '__main__':

#     _, score_model_path, score_word_index, verify_model_path, verify_word_index = argv
#     score_model = load_model(score_model_path)
#     verify_model = load_model(verify_model_path)

#     score_tokenizer = Tokenizer()
#     with open(score_word_index) as f:
#         score_tokenizer.word_index = json.load(f)

#     verify_tokenizer = Tokenizer()
#     with open(verify_word_index) as f:
#         verify_tokenizer.word_index = json.load(f)

#     claims = ['Fox 2000 Pictures released the film Soul Food.',
#               'Fox 2000 Pictures released the film Soul Food.']
#     evidences = ['Soul food is a type of cuisine .',
#                  'Soul Food is a 1997 American comedy-drama film produced by Kenneth `` Babyface '' Edmonds , Tracey Edmonds and Robert Teitel and released by Fox 2000 Pictures .']
#     score_preds = general_predict(score_model, score_tokenizer, claims, evidences)
#     print (score_preds)

#     verify_preds = general_predict(verify_model, verify_tokenizer, claims, evidences)
#     print(verify_preds)
