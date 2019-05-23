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
                        if len(support) < sample_amount:
                            for evidence in doc['evidence']:
                                _, content = self._getDoc(evidence)
                                refute.append([index, id, doc['claim'], doc['label'], content.strip()])
                                index += 1
                                if index % 100 == 0:
                                    print('%d examples loaded' % index)
                    else:
                        if len(support) < sample_amount:
                            _, content = self._searchDocs(doc['claim'])
                            no_info.append([index, id, doc['claim'], doc['label'], content[0].strip()])
                            index += 1
                            if index % 100 == 0:
                                print('%d examples loaded' % index)
                    if len(support) == sample_amount and len(refute) == sample_amount and len(no_info) == sample_amount:
                        break
                except Exception:
                    continue
        
        total = support + refute + no_info
        df = pd.DataFrame(total, columns=['index', 'id', 'label', 'claim', 'evidence']).sample(frac=1).reset_index(drop=True)
        
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

                            
    def train_dev_loader(self, is_trainset=True, max_sample=float('inf')):
        if is_trainset:
            dir = os.path.join(self.datadir, 'train.json')
        else:
            dir = os.path.join(self.datadir, 'devset.json')

        with open(dir) as f:
            data = json.loads(f.read())
            sup_examples, ref_examples, nei_examples = [], [], []
            c = 0
            for i, d in list(data.items()):
                try:
                    if d['label'] == 'SUPPORTS':
                        if len(sup_examples) < max_sample:
                            for e in d['evidence']:
                                _, content = self._retrieve(e)
                                sup_examples.append([c, i, d['claim'], content.strip(), d['label']])
                                c += 1
                                if c % 50 == 0:
                                    print('%d examples loaded' % c)
                    elif d['label'] == 'REFUTES':
                        if len(ref_examples) < max_sample:
                            for e in d['evidence']:
                                _, content = self._retrieve(e)
                                ref_examples.append([c, i, d['claim'], content.strip(), d['label']])
                                c += 1
                                if c % 50 == 0:
                                    print('%d examples loaded' % c)
                    else:
                        if len(nei_examples) < max_sample:
                            _, content = self._search(d['claim'])
                            content = content[0]
                            nei_examples.append([c, i, d['claim'], content.strip(), d['label']])
                            c += 1
                            if c % 50 == 0:
                                print('%d examples loaded' % c)
                    if len(sup_examples) == max_sample and len(nei_examples) == max_sample \
                        and len(ref_examples) == max_sample:
                        break
                except Exception:
                    continue
        samples = sup_examples + ref_examples + nei_examples
        
        df = pd.DataFrame(samples, columns=['index', 'id', 'claim', 'evidence', 'label'])

        return df.sample(frac=1).reset_index(drop=True)

    def test_loader(self):
        dir = os.path.join(self.datadir, 'test-unlabelled.json')
        with open(dir) as f:
            data = json.loads(f.read())
            examples = []
            c = 0
            for i, d in list(data.items())[:5]:
                claim = d['claim']
                docnames, contents = self._search(claim)
                for j in range(len(docnames)):
                    examples.append([c, i, claim, docnames[j], contents[j].strip()])
                    c += 1
        return pd.DataFrame(examples, columns=['index', 'id', 'claim', 'docname', 'evidence'])

    def _getDoc(self, e):
        doc, sentense_id = e[0], e[1]
        return self.search_engine.retrieve(doc, sentense_id)

    def _searchDocs(self, q):
        return self.search_engine.searchDocs(q)

