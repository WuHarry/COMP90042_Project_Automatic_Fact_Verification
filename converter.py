import os
import pandas as pd
import unicodedata
import json
import pickle
from search_engine import IndexFiles


class Converter(object):

    def __init__(self, root, storedir, datadir, datasetdir):
        self.datadir = datadir
        self.dataset_dir = datasetdir
        self.search_engine = IndexFiles(root, storedir)
    
    def training_data_converter(self, sample_amount=float('inf'), is_training=True):
        if is_training:
            set_dir = os.path.join(self.datadir, 'train.json')
        else:
            set_dir = os.path.join(self.datadir, 'devset.json')

        with open(set_dir) as f:
            dataset = json.loads(f.read())
            support, refute, no_info = [], [], []
            index = 0
            print(len(dataset.items()))
            for id, doc in dataset.items():
                try:
                    if doc['label'] == 'SUPPORTS':
                        if len(support) < sample_amount:
                            for evidence in doc['evidence']:
                                _, content = self._getDoc(evidence)
                                support.append([index, id, doc['claim'], doc['label'], content.strip()])
                                index += 1
                                if index % 100 == 0:
                                    print('%d examples loaded' % index)
                    elif doc['label'] == 'REFUTES':
                        if len(refute) < sample_amount:
                            for evidence in doc['evidence']:
                                _, content = self._getDoc(evidence)
                                refute.append([index, id, doc['claim'], doc['label'], content.strip()])
                                index += 1
                                if index % 100 == 0:
                                    print('%d examples loaded' % index)
                    else:
                        if len(no_info) < sample_amount:
                            _, content = self._searchDocs(doc['claim'])
                            no_info.append([index, id, doc['claim'], doc['label'], content[0].strip()])
                            index += 1
                            if index % 100 == 0:
                                print('%d examples loaded' % index)
                    if len(support) == sample_amount and len(refute) == sample_amount and len(no_info) == sample_amount:
                        print(len(support), len(refute), len(no_info))
                        break
                except Exception:
                    continue
        
        total = support + refute + no_info
        df = pd.DataFrame(total, columns=['index', 'id', 'claim', 'label', 'evidence']).sample(frac=1).reset_index(drop=True)
        
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir) 
        
        if is_training:
            data_set_path = os.path.join(self.dataset_dir, 'train%s.txt' % sample_amount)
        else:
            data_set_path = os.path.join(self.dataset_dir, 'dev%s.txt' % sample_amount)
        
        if not os.path.exists(data_set_path):
            with open(data_set_path, 'wb') as f:
                pickle.dump(df, f)
            print('Data set convertered!')
        else:
            print('Data already convertered!')
        
        return df

    def test_data_converter(self):
        test_dir = os.path.join(self.datadir, 'test-unlabelled.json')
        with open(test_dir) as f:
            data = json.loads(f.read())
            examples = []
            index = 0
            for id, doc in list(data.items())[:5]:
                claim = doc['claim']
                docnames, contents = self._searchDocs(claim)
                for j in range(len(docnames)):
                    examples.append([index, id, claim, docnames[j], contents[j].strip()])
                    index += 1

        df = pd.DataFrame(examples, columns=['index', 'id', 'claim', 'docname', 'evidence'])
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir) 
        test_set_path = os.path.join(self.dataset_dir, 'test.txt')
        if not os.path.exists(test_set_path):
            with open(test_set_path, 'wb') as f:
                pickle.dump(df, f)

        return df

    def _getDoc(self, e):
        doc, sentense_id = e[0], e[1]
        return self.search_engine.getDoc(doc, sentense_id)

    def _searchDocs(self, q):
        return self.search_engine.searchDocs(q)

