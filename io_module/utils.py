__author__ = 'max'

import numpy as np
import torch
from collections import OrderedDict
import pickle
from gensim.models.word2vec import Word2Vec
import gzip

from io_module.common import DIGIT_RE


def get_batch(data, batch_size, unk_replace=0.):
    data, data_size = data
    batch_size = min(data_size, batch_size)
    index = torch.randperm(data_size).long()[:batch_size]

    lengths = data['LENGTH'][index]
    max_length = lengths.max().item()
    words = data['WORD']
    single = data['SINGLE']
    words = words[index, :max_length]
    single = single[index, :max_length]
    if unk_replace:
        ones = single.new_ones(batch_size, max_length)
        noise = single.new_empty(batch_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    stack_keys = set(stack_keys)
    batch = {'WORD': words, 'LENGTH': lengths}
    batch.update({key: field[index, :max_length] for key, field in data.items() if key not in exclude_keys})
    batch.update({key: field[index, :2 * max_length - 1] for key, field in data.items() if key in stack_keys})
    return batch


def get_bucketed_batch(data, batch_size, unk_replace=0.):
    data_buckets, bucket_sizes = data
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in range(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in range(len(buckets_scale)) if buckets_scale[i] > random_number])

    data = data_buckets[bucket_id]
    bucket_size = bucket_sizes[bucket_id]
    batch_size = min(bucket_size, batch_size)
    index = torch.randperm(bucket_size).long()[:batch_size]

    lengths = data['LENGTH'][index]
    max_length = lengths.max().item()
    words = data['WORD']
    single = data['SINGLE']
    words = words[index, :max_length]
    single = single[index, :max_length]
    if unk_replace:
        ones = single.new_ones(batch_size, max_length)
        noise = single.new_empty(batch_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH'] + stack_keys)
    stack_keys = set(stack_keys)
    batch = {'WORD': words, 'LENGTH': lengths}
    batch.update({key: field[index, :max_length] for key, field in data.items() if key not in exclude_keys})
    batch.update({key: field[index, :2 * max_length - 1] for key, field in data.items() if key in stack_keys})
    return batch


def iterate_batch(data, batch_size, unk_replace=0., shuffle=False):
    data, data_size = data

    words = data['WORD']
    single = data['SINGLE']
    max_length = words.size(1)
    if unk_replace:
        ones = single.new_ones(data_size, max_length)
        noise = single.new_empty(data_size, max_length).bernoulli_(unk_replace).long()
        words = words * (ones - single * noise)

    indices = None
    if shuffle:
        indices = torch.randperm(data_size).long()
        indices = indices.to(words.device)

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'LAB'] + stack_keys)
    stack_keys = set(stack_keys)
    for start_idx in range(0, data_size, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        lengths = data['LENGTH'][excerpt]
        batch_length = lengths.max().item()
        batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
        # ugly fix the LAB
        if 'LAB' in data:
            batch['LAB'] = data['LAB'][excerpt]
        batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
        batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
        yield batch


def iterate_bucketed_batch(data, batch_size, unk_replace=0., shuffle=False):
    data_tensor, bucket_sizes = data

    bucket_indices = np.arange(len(bucket_sizes))
    if shuffle:
        np.random.shuffle((bucket_indices))

    stack_keys = ['STACK_HEAD', 'CHILD', 'SIBLING', 'STACK_TYPE', 'SKIP_CONNECT', 'MASK_DEC']
    exclude_keys = set(['SINGLE', 'WORD', 'LENGTH', 'LAB'] + stack_keys)
    stack_keys = set(stack_keys)
    for bucket_id in bucket_indices:
        data = data_tensor[bucket_id]
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        words = data['WORD']
        single = data['SINGLE']
        bucket_length = words.size(1)
        if unk_replace:
            ones = single.new_ones(bucket_size, bucket_length)
            noise = single.new_empty(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(words.device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            lengths = data['LENGTH'][excerpt]
            batch_length = lengths.max().item()
            batch = {'WORD': words[excerpt, :batch_length], 'LENGTH': lengths}
            # ugly fix the LAB
            if 'LAB' in data:
                batch['LAB'] = data['LAB'][excerpt]
            batch.update({key: field[excerpt, :batch_length] for key, field in data.items() if key not in exclude_keys})
            batch.update({key: field[excerpt, :2 * batch_length - 1] for key, field in data.items() if key in stack_keys})
            yield batch


def iterate_data(data, batch_size, bucketed=False, unk_replace=0., shuffle=False):
    if bucketed:
        return iterate_bucketed_batch(data, batch_size, unk_replace==unk_replace, shuffle=shuffle)
    else:
        return iterate_batch(data, batch_size, unk_replace==unk_replace, shuffle=shuffle)


def load_embedding_dict(embedding, embedding_path, normalize_digits=True):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :return: embedding dict, embedding dimention, caseless
    """
    print("loading embedding: %s from %s" % (embedding, embedding_path))
    if embedding == 'word2vec':
        # loading word2vec
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        # loading GloVe
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt', encoding='utf8') as file:
            for line1 in file:
                line = line1.strip()
                if len(line) == 0:
                    continue

                tokens = line.split(' ')
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                elif embedd_dim +1 != len(tokens):
                    continue
                    # assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        # loading Senna
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=np.float32)
                embedd[:] = tokens[1:]
                word = DIGIT_RE.sub("0", tokens[0]) if normalize_digits else tokens[0]
                embedd_dict[word] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'sskip':
        embedd_dim = -1
        embedd_dict = OrderedDict()
        with gzip.open(embedding_path, 'rt') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                try:
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    if len(tokens) < embedd_dim:
                        continue

                    if embedd_dim < 0:
                        embedd_dim = len(tokens) - 1

                    embedd = np.empty([1, embedd_dim], dtype=np.float32)
                    start = len(tokens) - embedd_dim
                    word = ' '.join(tokens[0:start])
                    embedd[:] = tokens[start:]
                    word = DIGIT_RE.sub("0", word) if normalize_digits else word
                    embedd_dict[word] = embedd
                except UnicodeDecodeError:
                    continue
        return embedd_dict, embedd_dim
    elif embedding == 'polyglot':
        words, embeddings = pickle.load(open(embedding_path, 'rb'), encoding='latin1')
        _, embedd_dim = embeddings.shape
        embedd_dict = OrderedDict()
        for i, word in enumerate(words):
            embedd = np.empty([1, embedd_dim], dtype=np.float32)
            embedd[:] = embeddings[i, :]
            word = DIGIT_RE.sub("0", word) if normalize_digits else word
            embedd_dict[word] = embedd
        return embedd_dict, embedd_dim

    else:
        raise ValueError("embedding should choose from [word2vec, senna, glove, sskip, polyglot]")