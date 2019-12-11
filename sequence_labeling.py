# coding: utf-8
import argparse
import math
import os
import random
from typing import List

import torch.optim as optim
from tqdm import tqdm

from io_module.data_loader import *
from io_module.logger import get_logger
from model.sequence_labeling import *


# out is [batch, max_len+2]
def standardize_batch(sample_list: List) -> (torch.Tensor, torch.Tensor):
    max_len = max([len(sentence[0]) for sentence in sample_list])

    standardized_sentence_list = []
    standardized_label_list = []
    revert_idx_list = []
    mask_list = []
    for sample in sample_list:
        sentence, label = sample
        standardized_sentence = sentence + [0] * (max_len - len(sentence))
        standardized_label = label + [0] * (max_len - len(label))
        mask = [1] * len(sentence) + [0] * (max_len - len(sentence))
        revert_idx = [i for i in range(len(sentence) + 2)][::-1] + [i for i in range(len(sentence) + 2, max_len + 2)]
        standardized_sentence_list.append(np.array(standardized_sentence))
        standardized_label_list.append(np.array(standardized_label))
        mask_list.append(np.array(mask))
        revert_idx_list.append(np.array(revert_idx))
    return torch.tensor(standardized_sentence_list).long(), torch.tensor(standardized_label_list).long(),\
           torch.tensor(mask_list).long(), torch.tensor(revert_idx_list).long()


def main():
    parser = argparse.ArgumentParser(description="Gaussian Input Output HMM")

    parser.add_argument(
        '--data',
        type=str,
        default='./dataset/syntic_data_yong/0-1000-10-new',
        help='location of the data corpus')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--var_scale', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=10)

    args = parser.parse_args()
    # np.random.seed(global_variables.RANDOM_SEED)
    # torch.manual_seed(global_variables.RANDOM_SEED)
    # random.seed(global_variables.RANDOM_SEED)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    epoch = args.epoch
    batch_size = args.batch
    lr = args.lr
    momentum = args.momentum
    root = args.data

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # TODO ntokens generate from dataset
    ntokens = 1000
    nlabels = 5
    # save parameter
    logger = get_logger('Sequence-Labeling')
    # logger = LOGGER
    logger.info(args)

    logger.info('Parameter From global_variables.py')
    logger.info('LOG_PATH:' + LOG_PATH)
    logger.info('EMISSION_CHO_GRAD:' + str(EMISSION_CHO_GRAD))
    logger.info('TRANSITION_CHO_GRAD:' + str(TRANSITION_CHO_GRAD))
    logger.info('DECODE_CHO_GRAD:' + str(DECODE_CHO_GRAD))
    logger.info('FAR_TRANSITION_MU:' + str(FAR_TRANSITION_MU))
    logger.info('FAR_DECODE_MU:' + str(FAR_DECODE_MU))
    logger.info('FAR_EMISSION_MU:' + str(FAR_EMISSION_MU))
    logger.info('RANDOM_SEED:' + str(args.random_seed))

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    print(torch.cuda.is_available())
    # Loading data
    logger.info('Load data....')
    train_dataset = sequence_labeling_data_loader(root, type='train')
    dev_dataset = sequence_labeling_data_loader(root, type='dev')
    test_dataset = sequence_labeling_data_loader(root, type='test')

    # build model
    model = MixtureGaussianSequenceLabeling(dim=args.dim, ntokens=ntokens, nlabels=nlabels)
    # model = RNNSequenceLabeling("RNN_TANH", ntokens=ntokens, nlabels=nlabels, ninp=10, nhid=10)
    model.to(device)
    logger.info('Building model ' + model.__class__.__name__ + '...')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # depend on dev ppl
    best_epoch = (-1, 0.0)
    for i in range(epoch):
        epoch_loss = 0
        random.shuffle(train_dataset)
        model.train()
        optimizer.zero_grad()
        for j in tqdm(range(math.ceil(len(train_dataset) / batch_size))):
            samples = train_dataset[j * batch_size: (j + 1) * batch_size]

            sentences, labels, masks, revert_order = standardize_batch(samples)
            loss = model.get_loss(sentences.to(device), labels.to(device), masks.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += (loss.item()) * sentences.size(0)
        logger.info('Epoch ' + str(i) + ' Loss: ' + str(round(epoch_loss / len(train_dataset), 4)))
        dev_sentences, dev_labels, dev_masks, dev_revert_order = standardize_batch(dev_dataset)
        model.eval()
        with torch.no_grad():
            acc, corr = model.get_acc(dev_sentences.to(device), dev_labels.to(device),
                                      dev_masks.to(device))
            logger.info("\t Dev Acc " + str(round(acc.item()*100, 2)))

        if best_epoch[1] < acc:
            best_epoch = (i, acc.item())
    logger.info("Best Epoch: " + str(best_epoch[0]) + " ACC: " + str(round(best_epoch[1], 5)))

if __name__ == '__main__':
    main()
