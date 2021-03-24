"""Read All Configuration from DataManager/DataConfig/*.yaml"""

import os
import yaml
import json


def read_schema(schema_conf_file):
    with open(schema_conf_file) as f:
        return {k: v for k, v in yaml.load(f, Loader=yaml.FullLoader).items()} # feature id -> field name

def read_field_conf(data_dir, schema_conf_file, field_conf_file):
    with open(field_conf_file) as f:
        valid_field_name = read_schema(schema_conf_file).values()
        field_conf = yaml.load(f, Loader=yaml.FullLoader)

        for field, conf in field_conf.items():
            check_field_conf(data_dir, field, **conf)
        
        # delete unused fields
        delete_fields = set(field_conf.keys()) - set(valid_field_name)

        for field in delete_fields:
            del field_conf[field]

        return field_conf


def check_field_conf(data_dir, field, **kwargs):
    type_ = kwargs["type"]
    trans = kwargs["transform"]
    param = kwargs["parameter"]
    missing_strategy = kwargs["missing_strategy"]
    feat_stat = kwargs["feat_stat"]

    if type_ is None:
        raise ValueError("Type are required in field conf, "
                            "found empty value for field `{}`".format(field))

    assert type_ in {'category', 'continuous'}, (
            "Invalid type `{}` for field `{}` in field conf, "
            "must be 'category' or 'continuous'".format(type_, field))
    
    # check transform and parameter
    if type_ == 'category':
        assert trans in {'hash_bucket', 'identity', 'vocab'}, (
                "Invalid transform `{}` for field `{}` in field conf, "
                "must be one of `hash_bucket`, `vocab`, `identity`.".format(trans, field))
        assert missing_strategy in {'most-frequent', 'special-token', 'zero-out'}, (
                "Invalid missing_strategy `{}` for field `{}` in field conf, "
                "must be one of `most-frequent`, `special-token`, and 'zero-out'.".format(missing_strategy, field))
        
        if not isinstance(os.path.join(data_dir, feat_stat), str) or not os.path.isfile(os.path.join(data_dir, feat_stat)):
            raise TypeError('Invalid feature statistics file path `{}` for field `{}` in field conf, '
                                'feat_stat must be a path to feature statistics file.'.format(os.path.join(data_dir, feat_stat), field))
            with open(os.path.join(data_dir, feat_stat), 'r') as f:
                stat_json = json.load(f)
                assert stat_json.keys() == {'most_freq'}, 'Invalid statistic info found in `{}` for field `{}`. Only `most_freq` is supported.'.format(feat_stat, field)

        if trans == 'hash_bucket' or trans == 'identity':
            if not isinstance(param, int):
                raise TypeError('Invalid parameter `{}` for field `{}` in field conf, '
                                    '{} parameter must be an integer.'.format(param, field, trans))
        elif trans == 'vocab':
            if not isinstance(os.path.join(data_dir, param), str) or not os.path.isfile(os.path.join(data_dir, param)):
                raise TypeError('Invalid parameter `{}` for field `{}` in field conf, '
                                    'vocab parameter must be a path to vocabulary file.'.format(os.path.join(data_dir, param), field))
    else:
        normalization, boundaries = param['normalization'], param['boundaries']
        assert missing_strategy in {'most-frequent', 'mean', 'median', 'zero-out'}, (
                "Invalid missing_strategy `{}` for field `{}` in field conf, "
                "must be one of `most-frequent`, `mean`, `median`, `zero-out`.".format(missing_strategy, field))
        if not isinstance(os.path.join(data_dir, feat_stat), str) or not os.path.isfile(os.path.join(data_dir, feat_stat)):
            raise TypeError('Invalid feature statistics file path `{}` for field `{}` in field conf, '
                                'feat_stat must be a path to feature statistics file.'.format(os.path.join(data_dir, feat_stat), field))
            with open(os.path.join(data_dir, feat_stat), 'r') as f:
                stat_json = json.load(f)
                assert stat_json.keys() == {'min', 'max', 'mean', 'std', 'median', 'most_freq'}, 'Invalid statistic info found in `{}` for field `{}`.'
                " Only 'min', 'max', 'mean', 'std', 'median' and 'most_freq' is supported.".format(feat_stat, field)

        if trans:
            assert trans in {'min_max', 'log', 'log_square', 'standard'}, \
                    "Invalid transform `{}` for field `{}` in field conf, " \
                    "continuous feature transform must be `min_max` or `log` or `log_square` or `standard`.".format(trans, field)
            if trans == 'min_max' or trans == 'standard':
                if not isinstance(normalization, (list, tuple)) or len(normalization) != 2:
                    raise TypeError('Invalid normalization parameter `{}` for field `{}` in field conf, '
                                        'must be 2 elements list for `min_max` or `standard` scaler.'.format(normalization, field))
                if trans == 'min_max':
                    min_, max_ = normalization
                    if not isinstance(min_, (float, int)) or not isinstance(max_, (float, int)):
                        raise TypeError('Invalid normalization parameter `{}` for field `{}` in field conf, '
                                        'list elements must be int or float.'.format(normalization, field))
                    assert min_ < max_, ('Invalid normalization parameter `{}` for field `{}` in field conf, '
                                         '[min, max] list elements must be min<max'.format(normalization, field))
                elif trans == 'standard':
                    mean, std = normalization
                    if not isinstance(mean, (float, int)):
                        raise TypeError('Invalid normalization parameter `{}` for field `{}` in field conf, '
                                        'parameter mean must be int or float.'.format(mean, field))
                    if not isinstance(std, (float, int)) or std <= 0:
                        raise TypeError('Invalid normalization parameter `{}` for field `{}` in field conf, '
                                            'parameter std must be a positive number.'.format(std, field))
        if boundaries:
            if not isinstance(boundaries, (tuple, list)):
                raise TypeError('Invalid parameter `{}` for field `{}` in field conf, '
                                    'discretize parameter must be a list.'.format(boundaries, field))
            else:
                for v in boundaries:
                    assert isinstance(v, (int, float)), \
                            "Invalid parameter `{}` for field `{}` in field conf, " \
                            "discretize parameter element must be integer or float.".format(boundaries, field)
                    