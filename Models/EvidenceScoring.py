from __future__ import print_function
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.losses import *


class RNNCNNClassifier(object):
    def __init__(self, pretrained, vocab_size, embedding_dim, max_seq_len):
        super(RNNCNNClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.embeddings = Embedding(vocab_size, embedding_dim, weights=[pretrained], input_length=max_seq_len,
                                    trainable=True)

    def build_model(self):
        input_claim = Input(shape=(self.max_seq_len,), name='claim_input')
        input_evidence = Input(shape=(self.max_seq_len,), name='evidence_input')

        claim_embs = self.embeddings(input_claim)
        evidence_embs = self.embeddings(input_evidence)

        claim_embs = Bidirectional(LSTM(550, return_sequences=True))(claim_embs)
        evidence_embs = Bidirectional(LSTM(550, return_sequences=True))(evidence_embs)

        claim_embs = Dropout(0.2)(claim_embs)
        evidence_embs = Dropout(0.2)(evidence_embs)

        embedding_layer = concatenate([claim_embs, evidence_embs])
        conv_blocks = []
        filters = [1, 2, 3]
        for fz in filters:
            conv = Conv1D(filters=128,
                          kernel_size=fz,
                          activation='relu',
                          padding='same',
                          strides=1)(embedding_layer)
            conv = GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        z = concatenate(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        z = Dense(500)(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = Dropout(0.4)(z)
        output = Dense(2, activation='softmax', activity_regularizer=regularizers.l2(1e-04))(z)
        model = Model(inputs=[input_claim, input_evidence], outputs=output)

        opti = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=binary_crossentropy,
                      optimizer=opti,
                      metrics=['acc'])

        return model
