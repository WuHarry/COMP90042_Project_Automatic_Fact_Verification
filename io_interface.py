import os
import pickle
from converter import Converter

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
                input_data = pickle.load(f)
                print("Dataset loaded!")
        else:
            input_data = self.converter.training_data_converter(sample_amount, is_training)  

        return input_data

    def generateTest(self, isFinal=True):
        if isFinal:
            test_set_path = os.path.join(self.dataset_dir, 'test.txt')
        else:
            test_set_path = os.path.join(self.dataset_dir, 'dev-test.txt')
        if os.path.exists(test_set_path):
            with open(test_set_path, 'rb') as f:
                test_input = pickle.load(f)
                print("Test Data loaded!")
        else:
            test_input = self.converter.test_data_converter(isFinal)

        return test_input



