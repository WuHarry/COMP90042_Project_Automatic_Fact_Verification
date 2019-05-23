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
    
    def generateInput(self, sample_amount, is_training=True):
        dataset_file_path = os.path.join(self.dataset_dir, 'train%s.txt' % sample_amount)
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'rb') as f:
                df = pickle.load(f)
        else:
            df = self.converter.training_data_converter(sample_amount=sample_amount)

        input_data = df.apply(lambda x: InputData(id=x['index'],
                                                  claim=x['claim'],
                                                  evidence=x['evidence'],
                                                  label=x['label']), axis=1)
        return input_data

    



