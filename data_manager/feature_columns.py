import collections
import hashlib 
import json

# Data instances containing missing feature values represented by `-1` for int and `''` for 
# string must be dropped and cannot be sent to these feature columns.

"""
Note that all these feature columns do not handle missing feature values in data. 
Users need to write their own logic to handle missing feature values before using
these columns. 
"""


class HashedCategoricalColumn(object):
    def __init__(self, name, hash_bucket_size, missing_strategy, missing_value, feat_stat, embed_dim, aggregate, dtype):
        super(HashCategoricalColumn, self).__init__()
        self.name = name 
        self.dtype = dtype
        self.hash_bucket_size = hash_bucket_size
        self.hashmap = {}  # input value -> hash code
        self.missing_strategy = missing_strategy
        self.missing_value = missing_value
        self.feat_stat = feat_stat
        self.embed_dim = embed_dim 
        self.aggregate = aggregate
    
    def dim(self):
        return self.hash_bucket_size
    
    def name(self):
        return self.name 
    
    def type(self):
        return self.dtype
    
    def transform_input_value(self, input_value):
        if input_value == self.missing_value: # missing
            if self.missing_strategy == 'most-frequent':
                input_value = self.feat_stat['most_freq']
            if self.missing_strategy == 'special-token':
                input_value = '#' 
            if self.missing_strategy == 'zero-out':
                return -1

        assert isinstance(input_value, (str, int)), "HashedCategoricalColumn: only str, int inputs are supported, but got {}".format(type(input_value))

        # fast lookup 
        if input_value in self.hashmap:
            return self.hashmap[input_value]
        
        # must use hash function from hashlib
        # the built-in hash function of Python offsets the hash with a random seed (set once at startup) to prevent attackers 
        # from tar-pitting your application by sending you keys designed to collide.
        # Thus, the hash value for the same key may be different in different python processes.
        # It means that the hash code of the same value is only guaranteed to be same within the same python
        # process. 
        # https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
        # One can set a fixed seed or disable the feature by setting the PYTHONHASHSEED environment variable; 
        # the default is random but one can set it to a fixed positive integer value, with 0 disabling the feature altogether.
        # If you were relying on the order of keys in a Python set, then don't. Python uses a hash table to implement these 
        # types and their order depends on the insertion and deletion history as well as the random hash seed.
        # https://www.programiz.com/python-programming/methods/built-in/hash
        # Since we need a stable hash implementation, we use the hashlib module; this implements cryptographic hash functions.
        hash_object = hashlib.sha1(str(input_value).encode('utf-8'))
        hex_dig = hash_object.hexdigest()
        sha_value = int(hex_dig, 16)

        hash_value = sha_value % self.hash_bucket_size

        assert hash_value < self.hash_bucket_size

        self.hashmap[input_value] = hash_value

        return hash_value


class VocabularyFileCategoricalColumn(object):
    def __init__(self, name, vocabulary_file, vocabulary_size, num_oov_buckets, default_value, missing_strategy, missing_value, feat_stat, embed_dim, aggregate, dtype):
        super(VocabularyFileCategoricalColumn, self).__init__()
        self.name = name 
        self.dtype = dtype
        self.vocabulary_size = vocabulary_size
        self.missing_strategy = missing_strategy
        self.missing_value = missing_value
        self.feat_stat = feat_stat
        self.embed_dim = embed_dim 
        self.aggregate = aggregate

        if num_oov_buckets > 0:
            self.num_oov_buckets = num_oov_buckets
            self.default_value = -1
        else:
            self.num_oov_buckets = 0
            self.default_value = default_value
        
        # create category2value mapper
        self.key_mapper = {}
        with open(vocabulary_file, 'r') as f:
            for i, line in enumerate(f):
                vals = line.strip().split('\t')
                self.key_mapper[vals[0]] = i 
        
    def dim(self):
        if self.num_oov_buckets > 0:
            return self.vocabulary_size + self.num_oov_buckets
        else:
            if self.default_value >= self.vocabulary_size or self.default_value < 0:
                return self.vocabulary_size + 1
            else:
                return self.vocabulary_size
    
    def name(self):
        return self.name 
    
    def type(self):
        return self.dtype

    def transform_input_value(self, input_value):
        if input_value == self.missing_value: # missing
            if self.missing_strategy == 'most-frequent':
                input_value = self.feat_stat['most_freq']
            if self.missing_strategy == 'special-token':
                input_value = '#' 
            if self.missing_strategy == 'zero-out':
                return -1

        assert isinstance(input_value, (str, int)), "VocabularyFileCategoricalColumn: only str and int inputs are supported, but got {}".format(
            type(input_value))

        if input_value in self.key_mapper:
            return self.key_mapper[input_value]
        
        if self.num_oov_buckets > 0:
            hash_object = hashlib.sha1(str(input_value).encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            sha_value = int(hex_dig, 16)

            hash_value = sha_value % self.num_oov_buckets + self.vocabulary_size

            assert hash_value >= self.vocabulary_size and hash_value < self.vocabulary_size + self.num_oov_buckets

            return hash_value
        else:
            return self.default_value


class IdentityCategoricalColumn(object):
    def __init__(self, name, num_buckets, missing_strategy, missing_value, feat_stat, default_value, embed_dim, aggregate, dtype):
        super(IdentityCategoricalColumn, self).__init__()
        self.name = name 
        self.dtype = dtype
        self.missing_strategy = missing_strategy
        self.missing_value = missing_value 
        self.feat_stat = feat_stat
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim 
        self.aggregate = aggregate

        assert default_value >= 0
        self.default_value = default_value
    
    def dim(self):
        if self.default_value is None:
            return self.num_buckets
        else:
            if self.default_value >= self.num_buckets or self.default_value < 0:
                return self.num_buckets + 1
    
    def name(self):
        return self.name 
    
    def type(self):
        return self.dtype
    
    def transform_input_value(self, input_value):
        if input_value == self.missing_value: # missing 
            if self.missing_strategy == 'most-frequent':
                input_value = self.feat_stat['most_freq']
            if self.missing_strategy == 'special-token':
                input_value = -1
            if self.missing_strategy == 'zero-out':
                return -1

        assert isinstance(input_value, int), "VocabularyFileCategoricalColumn: only int inputs are supported, but got {}".format(
            type(input_value))

        if self.default_value is None:
            if input_value < 0 or input_value >= self.num_buckets:
                raise ValueError('IdentityCategoricalColumn: default_value is not set but input value {} is outside [0, {})'.format(
                    input_value, self.num_buckets))
            else:
                return input_value 
        else:
            if input_value < 0 or input_value >= self.num_buckets:
                return self.default_value
            else:
                return input_value 


class NumericColumn(object):
    def __init__(self, name, missing_strategy, missing_value, feat_stat, normalizer_fn, boundaries, dtype):
        super(NumericColumn, self).__init__()
        self.name = name 
        self.dtype = dtype
        self.missing_strategy = missing_strategy
        self.missing_value = missing_value
        self.feat_stat = feat_stat 
        self.normalizer_fn = normalizer_fn
        self.boundaries = boundaries
    
    def dim(self):
        if self.boundaries is None:
            return 1
        else:
            return len(self.boundaries) + 1
    
    def name(self):
        return self.name 
    
    def type(self):
        return self.dtype
    
    def transform_input_value(self, input_value):
        if input_value == self.missing_value: # missing
            if self.missing_strategy == 'most-frequent':
                input_value = self.feat_stat['most_freq']
            if self.missing_strategy == 'min':
                input_value = self.feat_stat['min']
            if self.missing_strategy == 'max':
                input_value = self.feat_stat['max']
            if self.missing_strategy == 'mean':
                input_value = self.feat_stat['mean']
            if self.missing_strategy == 'median':
                input_value = self.feat_stat['median']
            if self.missing_strategy == 'zero-out':
                return 0.
        if self.normalizer_fn:
            input_value = self.normalizer_fn(input_value)
        
        if not self.boundaries:
            return float(input_value)
        else:
            return np.searchsorted(self.boundaries, input_value, side='left')

def categorical_column_with_hash_bucket(name, hash_bucket_size, missing_strategy, missing_value, feat_stat_path, embed_dim, aggregate):
    """Represents sparse feature where ids are set by hashing.

    Use this when your sparse features are in string or integer format, and you
    want to distribute your inputs into a finite number of buckets by hashing.
    output_id = Hash(input_feature_string) % bucket_size for string type input.
  
    For int type input, the value is converted to its string representation first
    and then hashed by the same formula.
  
    Args:
        name: A unique string identifying the input feature. It is used as the
            column name and the dictionary key for feature
            `Tensor` objects, and feature columns.
        hash_bucket_size: An int > 1. The number of buckets.

    Returns:
        A `HashedCategoricalColumn`.

    Raises:
        ValueError: `hash_bucket_size` is not greater than 1.
    """
    if hash_bucket_size is None:
        raise ValueError('hash_bucket_size must be set. ' 'name: {}'.format(name))

    if hash_bucket_size < 1:
        raise ValueError('hash_bucket_size must be at least 1. '
                            'hash_bucket_size: {}, name: {}'.format(
                            hash_bucket_size, name))
    
    assert isinstance(name, str), 'name is expected to be a string, but got {} with type {}'.format(name, type(name))

    with open(feat_stat_path, 'r') as f:
        feat_stat = json.load(f)

    return HashedCategoricalColumn(name, hash_bucket_size, missing_strategy, missing_value, feat_stat, embed_dim, aggregate, dtype='category')


def categorical_column_with_vocabulary_file(
        name, 
        vocabulary_file,
        missing_strategy,
        missing_value,
        feat_stat_path, 
        embed_dim,
        aggregate='sum',
        vocabulary_size=None, 
        default_value=None, 
        num_oov_buckets=0):
    """A `CategoricalColumn` with a vocabulary file.

    Use this when your inputs are in string or integer format, and you have a
    vocabulary file that maps each value to an integer ID. By default,
    out-of-vocabulary values are ignored. Use either (but not both) of
    `num_oov_buckets` and `default_value` to specify how to include
    out-of-vocabulary values.

    Example with `num_oov_buckets`:
    File `'/us/states.txt'` contains 50 lines, each with a 2-character U.S. state
    abbreviation. All inputs with values in that file are assigned an ID 0-49,
    corresponding to its line number. All other values are hashed and assigned an
    ID 50-54.

    ```python
    column = categorical_column_with_vocabulary_file(
      name='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
    ```

    Example with `default_value`:
    File `'/us/states.txt'` contains 51 lines - the first line is `'XX'`, and the
    other 50 each have a 2-character U.S. state abbreviation. Both a literal
    `'XX'` in input, and other values missing from the file, will be assigned
    ID 0. All others are assigned the corresponding line number 1-50.

    ```python
    column = categorical_column_with_vocabulary_file(
      name='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
      default_value=0)
    ```

    Args:
        name: A unique string identifying the input feature. It is used as the
            column name and the dictionary key for feature `Tensor` objects, 
            and feature columns.
        vocabulary_file: The vocabulary file name.
        vocabulary_size: Number of the elements in the vocabulary. This must be no
            greater than length of `vocabulary_file`, if less than length, later
            values are ignored. If None, it is set to the length of `vocabulary_file`.
        default_value: The integer ID value to return for out-of-vocabulary feature
            values, defaults to `vocabulary_size+1`. This can not be specified with a positive
            `num_oov_buckets`.
        num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
            buckets. All out-of-vocabulary inputs will be assigned IDs in the range 
            `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
            the input value. A positive `num_oov_buckets` can not be specified with
            `default_value`.
    
    Returns:
        A `VocabularyFileCategoricalColumn`.
    
    Raises:
        ValueError: `vocabulary_file` is missing or cannot be opened.
        ValueError: `vocabulary_size` is missing or < 1.
        ValueError: `num_oov_buckets` is a negative integer.
        ValueError: `num_oov_buckets` and `default_value` are both specified.
        ValueError: `dtype` is neither string nor integer.
    """
    if not vocabulary_file:
        raise ValueError('Missing vocabulary_file in {}.'.format(name))

    if vocabulary_size is None:
        with open(vocabulary_file, 'r') as f:
            vocabulary_size = sum(1 for _ in f)
        
        # print(
        #     'vocabulary_size = {} in {} is inferred from the number of elements '
        #     'in the vocabulary_file {}.'.format(vocabulary_size, name, vocabulary_file))
    
    if vocabulary_size < 1:
        raise ValueError('Invalid vocabulary_size in {}.'.format(name))
    
    if num_oov_buckets:
        if default_value is not None:
            raise ValueError(
                'Can\'t specify both num_oov_buckets and default_value in {}.'.format(name))
        
        if num_oov_buckets < 0:
            raise ValueError('Invalid num_oov_buckets {} in {}.'.format(num_oov_buckets, name))
    
    assert isinstance(name, str)

    with open(feat_stat_path, 'r') as f:
        feat_stat = json.load(f)

    return VocabularyFileCategoricalColumn(
        name=name,
        vocabulary_file=vocabulary_file,
        vocabulary_size=vocabulary_size,
        num_oov_buckets=0 if num_oov_buckets is None else num_oov_buckets,
        default_value=vocabulary_size if default_value is None else default_value,
        missing_strategy=missing_strategy,
        missing_value=missing_value,
        feat_stat=feat_stat,
        embed_dim=embed_dim,
        aggregate=aggregate,
        dtype='category')


def categorical_column_with_identity(name, num_buckets, missing_strategy, missing_value, feat_stat_path, embed_dim, aggregate, default_value=None):
    """A `CategoricalColumn` that returns identity values.

    Use this when your inputs are integers in the range `[0, num_buckets)`, and
    you want to use the input value itself as the categorical ID. Values outside
    this range will result in `default_value` if specified, otherwise it will
    fail.

    Typically, this is used for contiguous ranges of integer indexes, but
    it doesn't have to be. This might be inefficient (high space cost), however, 
    if many of IDs are unused. Consider `categorical_column_with_hash_bucket` in that case.

    In the following examples, each input in the range `[0, 1000000)` is assigned
    the same value. All other inputs are assigned `default_value` 0. Note that a
    literal 0 in inputs will result in the same default ID.

    ```python
    video_id = categorical_column_with_identity(
      name='video_id', num_buckets=1000000, default_value=0)
    ```

    Args:
        name: A unique string identifying the input feature. It is used as the
            column name and the dictionary key for feature `Tensor` objects, 
            and feature columns.
        num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
        default_value: If set, values outside of range `[0, num_buckets)` will
            be replaced with this value. If not set, values < 0 or values >= num_buckets will
            cause a failure.
    
    Returns:
        A `IdentityCategoricalColumn`
    
    Raises:
        ValueError: if `num_buckets` is less than one.
        ValueError: if `default_value` is not in range `[0, num_buckets)`.
    """
    if num_buckets < 1:
        raise ValueError(
            'num_buckets {} < 1, column_name {}'.format(num_buckets, name))
    
    if (default_value is not None) and (
        (default_value < 0) or (default_value >= num_buckets)):
        raise ValueError(
            'default_value {} not in range [0, {}), column_name {}'.format(
                default_value, num_buckets, name))
    
    assert isinstance(name, str)

    with open(feat_stat_path, 'r') as f:
        feat_stat = json.load(f)

    return IdentityCategoricalColumn(
        name=name, num_buckets=num_buckets, missing_strategy=missing_strategy, missing_value=missing_value, 
        feat_stat=feat_stat, default_value=default_value, embed_dim=embed_dim, aggregate=aggregate, dtype='category')

def numeric_column(name, missing_strategy, missing_value, feat_stat_path, normalizer_fn=None, boundaries=[]):
    """Represents real valued or numerical features.

    Args:
        name: A unique string identifying the input feature. It is used as the
            column name and the dictionary key for feature `Tensor` objects, and 
            feature columns.
        normalizer_fn: If not `None`, a function that can be used to normalize the
            value of the tensor after `default_value` is applied for parsing. Normalizer function 
            takes the input value as its argument, and returns the output value. (e.g. 
            lambda x: (x - 3.0) / 4.2). Please note that even though the most common use case of 
            this function is normalization, it can be used for any kind of transformations.
    
    Returns:
        A `NumericColumn`.
    """
    if normalizer_fn is not None and not callable(normalizer_fn):
        raise TypeError(
            'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))
    
    assert isinstance(name, str)

    with open(feat_stat_path, 'r') as f:
        feat_stat = json.load(f)

    return NumericColumn(
        name, 
        missing_strategy,
        missing_value,
        feat_stat,
        normalizer_fn=normalizer_fn,
        boundaries=boundaries,
        dtype='continuous')

