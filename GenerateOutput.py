import os, json
import numpy as np
import io_interface 
from COMP90042_Project_Automatic_Fact_Verification.Mains import predict
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from COMP90042_Project_Automatic_Fact_Verification.Utils.preprocessing import *
from COMP90042_Project_Automatic_Fact_Verification.Models.EvidenceScoring import *

THRESHOLD_REVELANT = 0.8
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
            # print(test['claim'])
            claims = [test['claim']] * len(docnames)
            id = test['id']
            result['claim'] = test['claim']
            count += 1
            if count % 100 == 0:
                print('%d Tests proceed' % count)
            # get if there is enough info
            # 0/1 mode
            # score_preds = predict.general_predict(score_model, score_tokenizer, claims, e_contents)
            # probability mode
            score_preds = predict.general_predict(score_model, score_tokenizer, claims, e_contents, False)
            # for 0/1 mode
            # if all(s == 0 for s in score_preds):
            # next line for probability mode
            if all(s[1] < THRESHOLD_REVELANT for s in score_preds):
                result['label'] = 'NOT ENOUGH INFO'
                result['evidence'] = []
                outputs[id] = result
            else:
                # for 0/1 mode
                # irrelevant = list(np.where(score_preds == 0)[0])
                # for probability mode
                irrelevant = []
                for i in range(len(score_preds)):
                    if score_preds[i][1] < THRESHOLD_REVELANT:
                        irrelevant.append(i)
                # remove all irrelevant
                for i in sorted(irrelevant, reverse=True):
                    del docnames[i]
                    del claims[i]
                    del e_contents[i]
                # check it is support or not
                # for 0/1 mode
                verify_preds = predict.general_predict(verify_model, verify_tokenizer, claims, e_contents)
                # for probability mode
                # verify_preds = predict.general_predict(verify_model, verify_tokenizer, claims, e_contents, False)
                # print(verify_preds)
                # for 0/1 mode
                if not all(s == 0 for s in verify_preds):
                # for probability mode
                # if not all(s[1] < THRESHOLD_REVELANT for s in verify_preds):
                    result['label'] = 'SUPPORTS'
                    result['evidence'] = []
                    # for 0/1 mode
                    evidences = list(np.where(verify_preds == 1)[0])
                    # for probability mode
                    # evidences = []
                    # for i in range(len(verify_preds)):
                    #     if verify_preds[i][1] >= THRESHOLD_REVELANT:
                    #         evidences.append(i)
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
                    # for 0/1 mode
                    evidences = list(np.where(verify_preds == 0)[0])
                    # for probability mode
                    # evidences = []
                    # for i in range(len(verify_preds)):
                    #     if verify_preds[i][1] < THRESHOLD_REVELANT:
                    #         evidences.append(i)
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
            output_result_path = os.path.join(self.output_path, 'dev-test%s.json' % THRESHOLD_REVELANT)
            
        print(output_result_path)
        
        with open(output_result_path, 'w') as outfile:
            json.dump(outputs, outfile, indent=2)

if __name__ == '__main__':
    output = GenerateOutput()
    output.generateOutput(False)