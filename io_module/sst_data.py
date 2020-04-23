__author__ = 'Ehaschia'

import os.path
from collections import defaultdict, OrderedDict

import numpy as np
import torch

from io_module.alphabet import Alphabet
from io_module.common import DIGIT_RE
from io_module.common import PAD, PAD_ID_WORD
from io_module.common import ROOT, END
from io_module.logger import get_logger
from io_module.reader import SSTReader

# Special vocabulary symbols - we always put them at the start.
_START_VOCAB = [PAD, ROOT, END]
NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 30, 50, 80]



def create_alphabets(alphabet_directory, train_path, data_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurrence=1, normalize_digits=True):

    def expand_vocab():
        vocab_set = set(vocab_list)
        for data_path in data_paths:
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    tokens = line.split('\t')[0].split(' ')

                    for token in tokens:
                        word = DIGIT_RE.sub("0", token) if normalize_digits else token

                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', singleton=True)
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)
        vocab = defaultdict(int)
        with open(train_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split('\t')[0].split(' ')
                for token in tokens:
                    word = DIGIT_RE.sub("0", token) if normalize_digits else token
                    vocab[word] += 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurrence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            assert isinstance(embedd_dict, OrderedDict)
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurrence

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("Total Singleton Size:  %d" % len(singletons))
        multi_vocab = [word for word in vocab_list if word in _START_VOCAB or vocab[word] > min_occurrence]
        logger.info("Total Vocabulary Size (w.o rare words): %d" % len(multi_vocab))

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        if data_paths is not None and embedd_dict is not None:
            expand_vocab()

        for word in vocab_list:
            if word in multi_vocab:
                word_alphabet.add(word)
            elif word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))
            else:
                raise ValueError("Error word: " + word)

        word_alphabet.save(alphabet_directory)
    else:
        word_alphabet.load(alphabet_directory)

    word_alphabet.close()
    logger.info("Word Alphabet Size (Singleton): %d (%d)" % (word_alphabet.size(), word_alphabet.singleton_size()))
    return word_alphabet


def read_data(source_path: str, word_alphabet: Alphabet, max_size=None, normalize_digits=True):
    data = []
    max_length = 0
    print('Reading data from %s' % source_path)
    counter = 0
    reader = SSTReader(source_path, word_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        sent = inst.sentence
        data.append([sent.word_ids, inst.pos_ids])
        if max_length < inst.length():
            max_length = inst.length()
        inst = reader.getNext(normalize_digits=normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)

    data_size = len(data)
    wid_inputs = np.empty([data_size, max_length], dtype=np.int64)
    lid_inputs = np.empty([data_size], dtype=np.int64)


    masks = np.zeros([data_size, max_length], dtype=np.float32)
    single = np.zeros([data_size, max_length], dtype=np.int64)
    lengths = np.empty(data_size, dtype=np.int64)

    for i, inst in enumerate(data):
        wids, lid = inst
        inst_size = len(wids)
        lengths[i] = inst_size
        # word ids
        wid_inputs[i, :inst_size] = wids
        wid_inputs[i, inst_size:] = PAD_ID_WORD
        # label id
        lid_inputs[i] = lid
        # masks
        masks[i, :inst_size] = 1.0
        for j, wid in enumerate(wids):
            if word_alphabet.is_singleton(wid):
                single[i, j] = 1

    words = torch.from_numpy(wid_inputs)
    labels = torch.from_numpy(lid_inputs)
    masks = torch.from_numpy(masks)
    single = torch.from_numpy(single)
    lengths = torch.from_numpy(lengths)

    data_tensor = {'WORD': words, 'LAB': labels, 'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
    return data_tensor, data_size


def read_bucketed_data(source_path: str, word_alphabet: Alphabet,
                       max_size=None, normalize_digits=True):
    data = [[] for _ in _buckets]
    print('Reading data from %s' % source_path)
    counter = 0
    reader = SSTReader(source_path, word_alphabet)
    inst = reader.getNext(normalize_digits=normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            print("reading data: %d" % counter)

        inst_size = inst.length()
        sent = inst.sentence
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([sent.word_ids, inst.pos_ids])
                break

        inst = reader.getNext(normalize_digits=normalize_digits)
    reader.close()
    print("Total number of data: %d" % counter)

    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]
    data_tensors = []
    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            data_tensors.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        lid_inputs = np.empty([bucket_size], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)
        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):
            wids, lid = inst
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            # label ids
            lid_inputs[i] = lid
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if word_alphabet.is_singleton(wid):
                    single[i, j] = 1

        words = torch.from_numpy(wid_inputs)
        labels = torch.from_numpy(lid_inputs)
        masks = torch.from_numpy(masks)
        single = torch.from_numpy(single)
        lengths = torch.from_numpy(lengths)

        data_tensor = {'WORD': words, 'LAB': labels, 'MASK': masks, 'SINGLE': single, 'LENGTH': lengths}
        data_tensors.append(data_tensor)
    return data_tensors, bucket_sizes