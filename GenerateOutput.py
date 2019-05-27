import os, json
import numpy as np
import io_interface 
from Mains import predict
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from Utils.preprocessing import *
from Models.EvidenceScoring import *

class GenerateOutput(object):

    def __init__(self):
        self.output_path='./output'
        self.score_model_path='./Data/scoring_model.h5'
        self.score_word_index='./Data/score_word_index.json'
        self.verify_model_path='./Data/verifying_model.h5'
        self.verify_word_index='./Data/verify_word_index.json'
        self.input_generator = io_interface.InputDataGenerator()

    def generateOutput(self, isFinal=True):
        tests = self.input_generator.generateTest(isFinal)

        score_model = load_model(self.score_model_path)
        verify_model = load_model(self.verify_model_path)

        score_tokenizer = Tokenizer()
        with open(self.score_word_index) as f:
            score_tokenizer.word_index = json.load(f)

        verify_tokenizer = Tokenizer()
        with open(self.verify_word_index) as f:
            verify_tokenizer.word_index = json.load(f)

        outputs = {}
        count = 0
        for test in tests:
            result = {}
            docnames = [e[1] for e in test['evidence']]
            e_contents = [e[0] for e in test['evidence']]
            print(test['claim'])
            claims = [test['claim']] * len(docnames)
            id = test['id']
            result['claim'] = test['claim']
            count += 1
            if count % 100 == 0:
                print('%d Tests proceed' % count)
            # get if there is enough info
            score_preds = predict.general_predict(score_model, score_tokenizer, claims, e_contents, False)
            if all(s[1] < 0.7 for s in score_preds):
                result['label'] = 'NOT ENOUGH INFO'
                result['evidence'] = []
                outputs[id] = result
            else:
                # check it is support or not
                verify_preds = predict.general_predict(verify_model, verify_tokenizer, claims, e_contents)
                # print(verify_preds)
                if not all(s == 0  for s in verify_preds):
                    result['label'] = 'SUPPORTS'
                    result['evidence'] = []
                    evidences = list(np.where(verify_preds == 1)[0])
                    # print(evidences)
                    for i in evidences:
                        doc_sec = docnames[i].split()
                        if len(doc_sec) == 1:
                            continue
                        result['evidence'].append([doc_sec[0], int(doc_sec[1])])
                    outputs[id] = result
                else:
                    result['label'] = 'REFUTES'
                    result['evidence'] = []
                    evidences = list(np.where(verify_preds == 0)[0])
                    # print(evidences)
                    for i in evidences:
                        doc_sec = docnames[i].split()
                        if len(doc_sec) == 1:
                            continue
                        result['evidence'].append([doc_sec[0], int(doc_sec[1])])
                    outputs[id] = result
        
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path) 

        if isFinal:
            output_result_path = os.path.join(self.output_path, 'testoutput.json')
        else:
            output_result_path = os.path.join(self.output_path, 'dev-test.json')
        
        with open(output_result_path, 'w') as outfile:
            json.dump(outputs, outfile, indent=2)

if __name__ == '__main__':
    output = GenerateOutput()
    output.generateOutput(False)