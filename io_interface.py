import os
import pickle
from converter import Converter
from 


class InputData(object):
    def __init__(self, id, claim, label=None, evidence=None):
        self.id = id
        self.claim = claim
        self.label = label
        self.evidence = evidence

class InputDataGenerator(object):
    def __init__(self):
        self.wiki_dir = 'wiki-pages-text.zip'
        self.index_dir = './IndexFiles.index'
        self.original_data_dir = './'
        self.dataset_dir = './datasets'
        self.converter = Converter(self.wiki_dir, self.index_dir, 
            self.original_data_dir, self.dataset_dir)
    
    def generateInput(self, sample_amount=float('inf'), is_training=True):
        if is_training:
            dataset_file_path = os.path.join(self.dataset_dir, 'train%s.txt' % sample_amount)
        else:
            dataset_file_path = os.path.join(self.dataset_dir, 'dev%s.txt' % sample_amount)
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'rb') as f:
                df = pickle.load(f)
                print("Dataset loaded!")
        else:
            df = self.converter.training_data_converter(sample_amount, is_training)

        input_data = df.apply(lambda x: InputData(id=x['index'],
                                                  claim=x['claim'],
                                                  evidence=x['evidence'],
                                                  label=x['label']), axis=1)   
        
        # for data in input_data:
        #     temp = []
        #     temp[id] = data.id
        #     temp[claim] = data.claim
        #     temp[evidence] = []
        #     temp[evidence].append((evidence, 1))
            
        #     _, fake_evidences = self.converter.search_engine.searchDocs(data.claim)
        #     true_evidences = [x['evidence'] for x in input_data.loc[input_data['claim'] == data.claim]]


        return input_data

    def generateTest(self):
        test_set_path = os.path.join(self.dataset_dir, 'test.txt')
        if os.path.exists(test_set_path):
            with open(test_set_path, 'rb') as f:
                df = pickle.load(f)
                print("Test Data loaded!")
        else:
            df = self.converter.test_data_converter()

        input_examples = df.apply(lambda x: InputExample(id=x['index'],
                                                         claim=x['claim'],
                                                         evidence=x['evidence'],
                                                         label='NOT ENOUGH INFO'), axis=1)
        return input_examples



