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
            print('%d Total Claims need to convert' % len(dataset.items()))
            for id, doc in dataset.items():
                try:
                    if doc['label'] == 'SUPPORTS':
                        if len(support) < sample_amount:
                            _, fake_evidences = self._searchDocs(doc['claim'], 10+len(doc['evidence']))
                            true_evidences = []
                            for e in doc['evidence']:
                                _, content = self._getDoc(e)
                                true_evidences.append(content)
                            for t_e in true_evidences:
                                temp = {}
                                temp['index'] = index                               
                                temp['id'] = id
                                temp['claim'] = doc['claim']
                                temp['label'] = doc['label']
                                temp['evidence'] = []
                                temp['evidence'].append((t_e.strip(), 1))
                                for f_e in fake_evidences:
                                    if len(temp['evidence']) >= 10:
                                        break
                                    else:
                                        if f_e not in true_evidences:
                                            temp['evidence'].append((f_e.strip(), 0))
                                support.append(temp)
                                index += 1
                                if index % 100 == 0:
                                    print('%d examples loaded' % index)
                    elif doc['label'] == 'REFUTES':
                        if len(refute) < sample_amount:
                            _, fake_evidences = self._searchDocs(doc['claim'], 10+len(doc['evidence']))
                            true_evidences = []
                            for e in doc['evidence']:
                                _, content = self._getDoc(e)
                                true_evidences.append(content)
                            for t_e in true_evidences:
                                temp = {}
                                temp['index'] = index
                                temp['id'] = id
                                temp['claim'] = doc['claim']
                                temp['label'] = doc['label']
                                temp['evidence'] = []
                                temp['evidence'].append((t_e.strip(), 1))
                                for f_e in fake_evidences:
                                    if len(temp['evidence']) >= 10:
                                        break
                                    else:
                                        if f_e not in true_evidences:
                                            temp['evidence'].append((f_e.strip(), 0))
                                refute.append(temp)
                                index += 1
                                if index % 100 == 0:
                                    print('%d examples loaded' % index)
                    else:
                        if len(no_info) < sample_amount:
                            _, content = self._searchDocs(doc['claim'], 10)
                            temp = {}
                            temp['index'] = index
                            temp['id'] = id
                            temp['claim'] = doc['claim']
                            temp['label'] = doc['label']
                            temp['evidence'] = []
                            for e in content:
                                if len(temp['evidence']) >= 10:
                                    break
                                else:
                                    temp['evidence'].append((e.strip(), 0))
                            no_info.append(temp)
                            index += 1
                            if index % 100 == 0:
                                print('%d examples loaded' % index)
                    if len(support) == sample_amount and len(refute) == sample_amount and len(no_info) == sample_amount:
                        print(len(support), len(refute), len(no_info))
                        break
                except Exception:
                    print('ERROR::::  exception: ', id, doc)
                    continue
        
        total = support + refute + no_info
        
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir) 
        
        if is_training:
            data_set_path = os.path.join(self.dataset_dir, 'train%s.txt' % sample_amount)
        else:
            data_set_path = os.path.join(self.dataset_dir, 'dev%s.txt' % sample_amount)
        
        if not os.path.exists(data_set_path):
            with open(data_set_path, 'wb') as f:
                pickle.dump(total, f)
            print('Data set convertered!')
        else:
            print('Data already convertered!')
        
        return total

    def test_data_converter(self, isFinal=True):
        if isFinal:
            test_dir = os.path.join(self.datadir, 'test-unlabelled.json')
        else:
            test_dir = os.path.join(self.datadir, 'devset.json')
        with open(test_dir) as f:
            data = json.loads(f.read())
            tests = []
            index = 0
            for id, doc in data.items():
                docnames, contents = self._searchDocs(doc['claim'], 20)
                temp = {}
                temp['index'] = index
                temp['id'] = id
                temp['claim'] = doc['claim']
                temp['label'] = 'UNKOWN'
                temp['evidence'] = []
                for i in range(len(docnames)):
                    temp['evidence'].append((contents[i].strip(), docnames[i]))
                tests.append(temp)
                index += 1
                if index % 100 == 0:
                    print('%d tests loaded' % index) 
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir) 
        if isFinal:
            test_set_path = os.path.join(self.dataset_dir, 'test.txt')
        else:
            test_set_path = os.path.join(self.dataset_dir, 'dev-test.txt')
        if not os.path.exists(test_set_path):
            with open(test_set_path, 'wb') as f:
                pickle.dump(tests, f)
            print('Data set convertered!')
        else:
            print('Data already convertered!')

        return tests

    def _getDoc(self, e):
        doc, sentense_id = e[0], e[1]
        return self.search_engine.getDoc(doc, sentense_id)

    def _searchDocs(self, q, topK=30):
        return self.search_engine.searchDocs(q, topK)

