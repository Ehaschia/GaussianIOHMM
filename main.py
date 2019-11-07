# coding: utf-8
import argparse
import math
from typing import List

from io_module.data_loader import *
from io_module.logger import get_logger
import numpy as np
from model.LM import GaussianBatchLanguageModel, RNNLanguageModel
import torch
import torch.optim as optim
import random
import datetime
import os
from tqdm import tqdm
import time
import global_variables


# perplexity calculator
def evaluate(dataset, model, device, batch_size=10, ntokens=10):
    model.eval()
    total_loss = 0.0
    total_length = 0
    with torch.no_grad():
        for j in tqdm(range(math.ceil(len(dataset) / batch_size))):
            samples = dataset[j * batch_size: (j + 1) * batch_size]
            batch_samples, masks = standardize_batch(samples, ntokens)
            batch_length = torch.sum(masks).item()
            total_loss += model.get_loss(batch_samples.to(device), masks.to(device)).item() * batch_length
            total_length += batch_length
    return total_loss / (total_length - 1)


# out is [batch, max_len+2]
def standardize_batch(sentence_list: List, ntokens=10) -> (torch.Tensor, torch.Tensor):
    max_len = max([len(sentence) for sentence in sentence_list])
    standardized_list = []
    mask_list = []
    for sentence in sentence_list:
        standardized_sentence = [ntokens] + sentence + [ntokens + 1] + [0] * (max_len - len(sentence))
        mask = [1] * (len(sentence) + 2) + [0] * (max_len - len(sentence))
        standardized_list.append(np.array(standardized_sentence))
        mask_list.append(mask)
    return torch.from_numpy(np.array(standardized_list)).long(), torch.from_numpy(np.array(mask_list)).long()


def main():
    parser = argparse.ArgumentParser(description="Gaussian Input Output HMM")

    parser.add_argument(
        '--data',
        type=str,
        default='E:/Code/GaussianIOHMM/dataset/hmm_generate_25/',
        help='location of the data corpus')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--var_scale', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=10)

    args = parser.parse_args()
    np.random.seed(global_variables.RANDOM_SEED)
    torch.manual_seed(global_variables.RANDOM_SEED)
    random.seed(global_variables.RANDOM_SEED)

    epoch = args.epoch
    batch_size = args.batch
    lr = args.lr
    momentum = args.momentum
    root = args.data

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # TODO ntokens generate from dataset
    ntokens = 50
    # save parameter
    logger = get_logger('IOHMM', global_variables.LOG_PATH)
    logger.info(args)

    logger.info('Parameter From global_variables.py')
    logger.info('LOG_PATH:' + global_variables.LOG_PATH)
    logger.info('EMISSION_CHO_GRAD:' + str(global_variables.EMISSION_CHO_GRAD))
    logger.info('TRANSITION_CHO_GRAD:' + str(global_variables.TRANSITION_CHO_GRAD))
    logger.info('DECODE_CHO_GRAD:' + str(global_variables.DECODE_CHO_GRAD))
    logger.info('FAR_TRANSITION_MU:' + str(global_variables.FAR_TRANSITION_MU))
    logger.info('FAR_DECODE_MU:' + str(global_variables.FAR_DECODE_MU))
    logger.info('FAR_EMISSION_MU:' + str(global_variables.FAR_EMISSION_MU))
    logger.info('RANDOM_SEED:' + str(global_variables.RANDOM_SEED))

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    print(torch.cuda.is_available())
    # Loading data
    logger.info('Load data....')
    train_dataset = hmm_generate_data_loader(root, type='train')
    dev_dataset = hmm_generate_data_loader(root, type='dev')
    test_dataset = hmm_generate_data_loader(root, type='test')

    # build model
    logger.info("Building model....")
    model = GaussianBatchLanguageModel(dim=args.dim, ntokens=ntokens)
    # model = RNNLanguageModel("RNN_TANH", ntokens=ntokens, ninp=10, nhid=10)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    epoch_loss_list = []

    for i in range(epoch):
        epoch_loss = 0.0
        total_length = 0
        random.shuffle(train_dataset)
        model.train()
        optimizer.zero_grad()
        for j in tqdm(range(math.ceil(len(train_dataset) / batch_size))):
            samples = train_dataset[j * batch_size: (j + 1) * batch_size]

            input_sample, mask = standardize_batch(samples)
            batch_length = torch.sum(mask).item()
            total_length += batch_length
            loss = model.get_loss(input_sample.to(device), mask.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * batch_length
        epoch_loss = epoch_loss / (total_length - 1)
        time.sleep(0.5)
        logger.info("Epoch:\t" + str(i) + "\t Training loss:\t" + str(round(epoch_loss, 4)) + "\t PPL: " + str(
            round(np.exp(epoch_loss), 4)))
        epoch_loss_list.append(epoch_loss)
        # evaluate
        dev_loss = evaluate(dev_dataset, model, device, ntokens=ntokens)
        test_loss = evaluate(test_dataset, model, device, ntokens=ntokens)
        time.sleep(0.5)
        logger.info("\t\t Dev Loss: " + str(round(dev_loss, 4)) + "\t PPL: " + str(round(np.exp(dev_loss), 4)))
        logger.info("\t\t Test Loss: " + str(round(test_loss, 4)) + "\t PPL: " + str(round(np.exp(test_loss), 4)))
        total_dev, masks = standardize_batch(dev_dataset, ntokens=ntokens)
        predict, corr_cnt, corr_acc = model.inference(total_dev, masks)
        logger.info("\t\t Dev Correct Number " + str(corr_cnt) + "\t Correct Acc: " + str(round(corr_acc, 4)))


if __name__ == '__main__':
    main()
    exit(1)
