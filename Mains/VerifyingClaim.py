from __future__ import print_function

from sys import argv

import tensorflow as tf

import os
import sys
sys.path.insert(0, os.path.abspath(".."))

from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping

from Utils.preprocessing import *
from Models.EvidenceScoring import *
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


class VerifyingClaim:
    def __init__(self, glove_path, train_data_path):
        super(VerifyingClaim, self).__init__()

        word2emb = build_index_from_glove(glove_path)

        train_pickle = read_train_set(train_data_path)

        claim_sequences, evidence_sequences, labels, word_index = preprocess_train_data_for_verifyclaim(train_pickle)
        with open('verify_word_index.json', 'w') as outfile:
            json.dump(word_index, outfile)
        print ('save word_index done')
        padded_claims = pad_sequences(claim_sequences, maxlen=50)
        del claim_sequences
        padded_evidence = pad_sequences(evidence_sequences, maxlen=50)
        del evidence_sequences
        train_indices, vali_indices = split_train_vali(padded_claims.shape[0], 0.05)

        self.train_data_claim = padded_claims[train_indices]
        self.train_data_evidence = padded_evidence[train_indices]
        self.train_label = labels[train_indices]

        self.vali_data_claim = padded_claims[vali_indices]
        self.vali_data_evidence = padded_evidence[vali_indices]
        self.vali_label = labels[vali_indices]

        del padded_evidence, padded_claims

        self.embedding = build_embedding_matrix(word_index, word2emb)

        self.vocab_size = self.embedding.shape[0]
        self.embedding_dim = self.embedding.shape[1]
        self.max_seq_len = 50

        self.classifier = RNNCNNClassifier(self.embedding, self.vocab_size, self.embedding_dim, self.max_seq_len)
        self.model = None

    def train(self):
        self.model = self.classifier.build_model()

        earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        callbacks_list = [earlystop]
        # Fit the model
        print('*'*80)
        print ("Model fiting")
        print('*'*80)

        self.model.fit([self.train_data_claim, self.train_data_evidence], self.train_label, validation_data=([self.vali_data_claim, self.vali_data_evidence], self.vali_label), epochs=30, batch_size=600, callbacks=callbacks_list, verbose=1)#callbacks=callbacks_list, verbose=0

        self.model.save('verifying_model.h5')




    def eval(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    _, glove_path, train_data_path = argv
    print(glove_path)
    print(train_data_path)
    sc = VerifyingClaim(glove_path, train_data_path)
    sc.train()
