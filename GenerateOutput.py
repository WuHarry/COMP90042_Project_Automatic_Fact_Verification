import os, json
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
        for test in tests[:10]:
            result = {}
            docnames = [e[1] for e in test['evidence']]
            e_contents = [e[0] for e in test['evidence']]
            print(test['claim'])
            claims = [test['claim']] * len(docnames)
            id = test['id']
            result['claim'] = test['claim']
            # get if there is enough info
            score_preds = predict.general_predict(score_model, score_tokenizer, claims, e_contents)
            print(score_preds)
            if all(s == 0 for s in score_preds):
                result['label'] = 'NOT ENOUGH INFO'
                result['evidence'] = []
                outputs[id] = result
                print(result)
                continue
            # check it is support or not
            verify_preds = predict.general_predict(verify_model, verify_tokenizer, claims, e_contents, False)

            # print(verify_preds)

            # result['label'] = label
            # result['evidence'] = []
            # for e in evidences:
            #     doc_sec = e.split()
            #     result['evidence'].append([doc_sec[0], doc_sec[1]])
            # result[id] = result
            # results.append(result)
        
        
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path) 

        if isFinal:
            output_result_path = os.path.join(self.output_path, 'test.json')
        else:
            output_result_path = os.path.join(self.output_path, 'dev-test.json')
        
        with open(output_result_path, 'w') as outfile:
            json.dump(outputs, outfile, indent=2)
        


if __name__ == '__main__':
    output = GenerateOutput()
    output.generateOutput(False)