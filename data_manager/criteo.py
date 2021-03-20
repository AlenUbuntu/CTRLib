import os 
import yaml
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import lmdb 
import struct
import json
import random

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict, Counter
from data_manager.read_conf import *
from data_manager.feature_columns import categorical_column_with_hash_bucket, categorical_column_with_identity, categorical_column_with_vocabulary_file, numeric_column


def analyze_criteo(path_to_data_file, max_records_to_read=int(1e7), min_frequency=10):
    dir_idx = path_to_data_file.rfind('/')
    if dir_idx == -1:
        dir_name = './'
    else:
        dir_name = path_to_data_file[:dir_idx]

    feature_type = {}

    for i in range(13):
        feature_type[i] = 'continuous'
    
    for i in range(13, 39):
        feature_type[i] = 'category'

        
    features = {}
    label_count = {}
    with open(path_to_data_file, 'r') as f:
        pbar = tqdm(f, mininterval=1, smoothing=0.1)

        count = 0
        for line in pbar:
            values = line.rstrip('\n').split('\t')

            if len(values) != 39 + 1:
                    continue
                
            label = int(values[0])
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1

            instance_features = values[1:]

            for i in range(len(instance_features)):
                if feature_type[i] == 'continuous':
                    if i not in features:
                        features[i] = {
                            'missing': 0,
                            'count': 0,
                            'values': []
                        }
                    if not instance_features[i]: # missing feature, skip
                        features[i]['missing'] += 1
                        features[i]['count'] += 1
                        continue
                    features[i]['count'] += 1
                    features[i]['values'].append(float(instance_features[i]))
                elif feature_type[i] == 'category':
                    # vocab
                    if i not in features:
                        features[i] = {
                            'missing': 0,
                            'count': 0,
                            'vocab': defaultdict(int)
                        }
                    if not instance_features[i]: # missing feature, skip
                        features[i]['missing'] += 1
                        features[i]['count'] += 1
                        continue
                    features[i]['count'] += 1
                    features[i]['vocab'][instance_features[i]] += 1
                else:
                    raise TypeError("Invalid feature type: {}".format(feature_type[i]))

            count += 1
            if count == max_records_to_read:
                break
    
    for i in feature_type:
        print("feature {}".format(i))
        if feature_type[i] == 'continuous':
            min_val = np.min(features[i]['values'])
            max_val = np.max(features[i]['values'])
            mean_val = np.mean(features[i]['values'])
            std_val = np.std(features[i]['values'])
            median_val = np.median(features[i]['values'])
            most_freq_val = sorted(Counter(features[i]['values']).items(), key=lambda x: x[1], reverse=True)[0][0]
            store_dict = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'most_freq': most_freq_val
            }
            print("\t Min: {}".format(min_val))
            print("\t Max: {}".format(max_val))
            print("\t Mean: {}".format(mean_val))
            print("\t Std: {}".format(std_val))
            print("\t Median: {}".format(median_val))
            print("\t Most Frequent: {}".format(most_freq_val))
            print("\t Missing Rate: {:.2f}%".format(features[i]['missing']/features[i]['count']*100))
            with open(os.path.join(dir_name, 'feature_{}_stat.json'.format(i)), 'w') as f:
                json.dump(store_dict, f)      
            fig, ax = plt.subplots(2, 1)
            data = np.array(features[i]['values'])
            copy_data = data.copy()
            copy_data[copy_data <= 0] = 1
            log2data = np.log2(copy_data)
            log2data[log2data==-np.inf] = 0
            ax[0].hist(data, bins=100)
            ax[1].hist(log2data, bins=100)
            fig.savefig('./DataManager/Criteo Feature {}.jpeg'.format(i))
        if feature_type[i] == 'category':
            most_freq_val = sorted(Counter(features[i]['vocab']).items(), key=lambda x: x[1], reverse=True)[0][0]
            store_dict = {
                'most_freq': most_freq_val
            }
            print('\t Vocab length: ', len(features[i]['vocab']))
            print("\t Most Frequent: {}".format(most_freq_val))
            print('\t Missing Rate: {:.2f}%'.format(features[i]['missing']/features[i]['count']*100))
            with open(os.path.join(dir_name, 'feat_{}.txt'.format(i)), 'w') as f:
                for key, val in sorted(features[i]['vocab'].items(), key=lambda x: x[1], reverse=True):
                    if val >= min_frequency:
                        f.write('{}\t{}\n'.format(key, val))
            with open(os.path.join(dir_name, 'feature_{}_stat.json'.format(i)), 'w') as f:
                json.dump(store_dict, f)      

    pprint(label_count) 


def get_numeric_transform(transform, *args):
    if transform == 'min_max':
        f = lambda x: (x-args[0])/(args[1]-args[0])
    elif transform == 'log':
        f = lambda x: np.log2(x) if x >= 1 else 0. # x >= 1
    elif transform == 'standard':
        f = lambda x: (x-args[0])/args[1]
    else:
        raise NotImplementedError("{} is not supported yet.".format(transform))
    
    return f 


def build_criteo(path_to_data_file, cache_path='./criteo', max_records_to_read=int(1e7)):
    with open(path_to_data_file, 'r') as f:
        pbar = tqdm(f, mininterval=1, smoothing=0.1)
        pbar.set_description('Create Criteo dataset cache: Setup LMDB Database!')

        labels = []
        index = 0
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            for line in pbar:
                values = line.rstrip('\n').split('\t')

                if len(values) != 39 + 1:
                    continue

                labels.append(int(values[0]))
                byte_code = line.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(struct.pack('>I', index), byte_code)

                index += 1

                if index == max_records_to_read:
                    break 
            labels = np.array(labels, dtype=np.uint32)
            with env.begin(write=True) as txn:
                txn.put(b'Labels', labels.tobytes())

        print("Create Criteo dataset cache: Done!")


class CriteoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg,  cache_path='./criteo'):
        super(CriteoDataset, self).__init__()
        config_dir = cfg.DATASET.CONFIG_DIR
        data_dir = cfg.DATASET.DIR
        schema_conf_file_path = os.path.join(config_dir, cfg.DATASET.SCHEMA_CONF_FILE)
        field_conf_file_path = os.path.join(config_dir, cfg.DATASET.FIELD_CONF_FILE)
        schema_conf = read_schema(schema_conf_file_path)  # {feature id: field_name}
        # self.schema_conf = {v:k for k, v in schema_conf.items()} # {field_name -> feature id}
        self.schema_conf = {} # {field_name -> [feature_id1, feature_id2, ...]}
        for feature_id, field_name in schema_conf.items():
            if field_name not in self.schema_conf:
                self.schema_conf[field_name] = []
            self.schema_conf[field_name].append(feature_id)
        
        self.field_conf = read_field_conf(data_dir, schema_conf_file_path, field_conf_file_path)  # field name -> conf
        self.n_folds = cfg.DATASET.N_FOLDS

        # build feature columns
        self.field_columns = {} # field name -> column

        for field_name, conf in self.field_conf.items():
            type_ = conf['type']
            trans_ = conf['transform']
            params_ = conf['parameter']
            missing_strategy = conf['missing_strategy']
            missing_value = conf['missing_value']
            feat_stat = conf['feat_stat']

            print('\n{}\t{}'.format(field_name, type_), end='')
            if type_ == 'category':
                if trans_ == 'hash_bucket':
                    column = categorical_column_with_hash_bucket(field_name, params_, missing_strategy, missing_value, os.path.join(data_dir, feat_stat))
                if trans_ == 'identity':
                    column = categorical_column_with_identity(field_name, params_, missing_strategy, missing_value, os.path.join(data_dir, feat_stat))
                if trans_ == 'vocab':
                    column = categorical_column_with_vocabulary_file(field_name, os.path.join(data_dir, params_), missing_strategy, missing_value, os.path.join(data_dir, feat_stat))
            else:
                normalization, boundaries = params_['normalization'], params_['boundaries']
                if trans_ == 'min_max':
                    min_, max_ = normalization
                    f = get_numeric_transform(trans_, min_, max_)
                    column = numeric_column(field_name, missing_strategy, missing_value, os.path.join(data_dir, feat_stat), f)
                if trans_ == 'standard':
                    mean, std = normalization
                    f = get_numeric_transform(trans_, mean, std)
                    column = numeric_column(field_name, missing_strategy, missing_value, os.path.join(data_dir, feat_stat), f)
                if trans_ == 'log':
                    f = get_numeric_transform(trans_)
                    column = numeric_column(field_name, missing_strategy, missing_value, os.path.join(data_dir, feat_stat), f)
            
            self.field_columns[field_name] = column 
        
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.database_size = txn.stat()['entries']-1
            self.labels = np.frombuffer(txn.get(b'Labels'), dtype=np.uint32)

        self.fold_size = self.database_size // self.n_folds
        self.idx = {}
    
    def set_split_fold(self, split='train', fold=0):
        assert fold >= 0 and fold < self.n_folds, 'Invalid setting of fold: {} as the total number of folds is {}'.format(fold, self.n_folds)

        idx = list(range(self.database_size))
        validation_start_idx = fold * self.fold_size 
        validation_end_idx = min(validation_start_idx + self.fold_size, self.database_size)
        validation_idx = idx[validation_start_idx:validation_end_idx]
        train_test_idx = idx[:validation_start_idx] + idx[validation_end_idx:]

        test_idx = train_test_idx[:2 * self.fold_size]
        train_idx = train_test_idx[2 * self.fold_size:]

        if split.lower() == 'train':
            self.idx = dict(zip(range(len(train_idx)), train_idx))
        elif split.lower() == 'valid':
            self.idx = dict(zip(range(len(validation_idx)), validation_idx))
        else:
            self.idx = dict(zip(range(len(test_idx)), test_idx))

    def get_sample_weights(self, label_weights):
        sample_weights = torch.zeros(len(self.idx))
        for index in self.idx:
            orig_idx = self.idx[index]
            label = self.labels[orig_idx]
            w = label_weights[label]
            sample_weights[index] = w
        return sample_weights.detach()

    def _transform(self, line):
        vals = line.rstrip('\n').split('\t')

        label, feats = int(vals[0]), vals[1:]

        data = {}

        for field_name in self.schema_conf:
            column = self.field_columns[field_name]

            if column.type() == 'category':
                indicator = torch.zeros(column.dim(), dtype=torch.long)
                valid_count = 0
                for fid in self.schema_conf[field_name]:
                    val = feats[fid-1]
                    if val != '':
                        index = column.transform_input_value(val)
                        if index >= 0 and index < column.dim():
                            indicator[index] = 1
                        valid_count += 1
                if valid_count == 0: # the entire field is empty
                    index = column.transform_input_value('')
                    if index >= 0 and index < column.dim():
                        indicator[index] = 1
                data[field_name] = indicator

            elif column.type() == 'continuous':
                assert len(self.schema_conf[field_name])==1, 'Continuous features must be itself an individual field! '
                'Multiple features with ids {} are found corresponding to field {}.'.format(self.schema_conf[field_name], field_name)
                fid = self.schema_conf[field_name][0]
                val = feats[fid-1]
                if val != '':
                    val = float(val)
                data[field_name] = torch.tensor([column.transform_input_value(val)])

        data['label'] = torch.tensor([label])
        return data
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, index):
        db_idx = self.idx[index]
        f = get_numeric_transform('log')
        with self.env.begin(write=False) as txn:
            line = txn.get(struct.pack('>I', db_idx)).decode('utf-8')
            data = self._transform(line)
        return data

if __name__ == '__main__':
    # analyze_criteo(
    #     path_to_data_file='/home/alan/Downloads/recommendation/Criteo/day_0',
    #     max_records_to_read=int(1e6)
    # )
    # build_criteo(
    #     path_to_data_file='/home/alan/Downloads/recommendation/Criteo/day_0',
    #     max_records_to_read=int(1e6)
    # )

    from configs import cfg 
    criteo = CriteoDataset(
        cfg
    )
    criteo.set_split_fold(
        split='test',
        fold=5
    )

    features_map = {}
    for i in range(39):
        if i < 13:
            with open('/home/alan/Downloads/recommendation/Criteo/feature_{}_stat.json'.format(i)) as f:
                features_map[i] = json.load(f)
        else:
            d = {}
            with open('/home/alan/Downloads/recommendation/Criteo/feat_{}.txt'.format(i)) as f:
                for j, line in enumerate(f):
                    cat = line.strip().split('\t')[0]
                    d[cat] = j 

            d['#'] = len(d)
            features_map[i] = d 

    f = get_numeric_transform('log')
    for i in range(len(criteo)):
        print("{}/{}".format(i+1, len(criteo)), end='\r')
        data, line = criteo[i]
        
        vals = line.strip().split('\t')

        for j, each in enumerate(vals[1:]):
            if j < 13:
                if each:
                    assert f(float(each)) == data['field_{}'.format(j+1)], '{}, {}'.format(f(float(each)), data['field_{}'.format(j+1)])
                else:
                    assert f(float(features_map[j]['most_freq'])) == data['field_{}'.format(j+1)], '{}, {}'.format(f(float(each)), data['field_{}'.format(j+1)])
            else:
                try:
                    if each in features_map[j]:
                        assert features_map[j][each] == torch.nonzero(data['field_{}'.format(j+1)], as_tuple=False), '{}, {}'.format(features_map[j][each], torch.nonzero(data['field_{}'.format(j+1)],as_tuple=False))
                    else:
                        assert features_map[j]['#'] == torch.nonzero(data['field_{}'.format(j+1)], as_tuple=False), '{}, {}'.format(features_map[j]['#'], torch.nonzero(data['field_{}'.format(j+1)],as_tuple=False))
                except RuntimeError:
                    print(vals)
                    print(j, torch.nonzero(data['field_{}'.format(j+1)], as_tuple=False), vals[j], features_map[j]['#'])
                    exit()
    print("Test completes successfully!")