# coding: utf-8
import argparse

from io_module.data_loader import *
from io_module.logger import get_logger
import numpy as np
from model.LM import GaussianLanguageModel
import torch
import torch.optim as optim
import random
import datetime
import os
from tqdm import tqdm
import time


def evaluate(dataset, model, device, ntokens=10):
    model.eval()
    total_loss = 0.0
    total_length = 0
    for j in tqdm(range(len(dataset))):
        a_sample = dataset[j]
        a_sample = [ntokens] + a_sample
        total_length += len(a_sample)
        input_sample = torch.from_numpy(np.array(a_sample)).long().to(device)
        loss = model.evaluate(input_sample)
        total_loss += loss
    return total_loss / (total_length-1)


def main():
    parser = argparse.ArgumentParser(description="Gaussian Input Output HMM")

    parser.add_argument('--data', type=str, default='E:/Code/GaussianIOHMM/dataset/hmm_generate',
                        help='location of the data corpus')

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--var_scale', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    epoch = args.epoch
    batch = args.batch
    lr = args.lr
    momentum = args.momentum
    root = args.data

    # TODO hard code
    ntokens = 10

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # save parameter
    logger = get_logger("IOHMM", args.log_dir)
    logger.info(args)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # Loading data
    logger.info('Load data....')
    train_dataset = data_loader(root, type='train') # syntic_data(root, type='train')
    dev_dataset = data_loader(root, type='dev') # syntic_data(root, type='dev')
    test_dataset = data_loader(root, type='test') # syntic_data(root, type='test')

    # bulid model
    logger.info("Building model....")
    model = GaussianLanguageModel(dim=args.dim, vocb_size=ntokens+1)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    epoch_loss_list = []

    for i in range(epoch):
        epoch_loss = 0.0
        total_length = 0
        random.shuffle(train_dataset)
        model.train()
        for j in tqdm(range(len(train_dataset))):
            a_sample = train_dataset[j]
            length = len(a_sample)
            total_length += length
            # padding zero:1000 and add <head>: 1001
            a_sample = [ntokens] + a_sample
            input_sample = torch.from_numpy(np.array(a_sample)).long().to(device)
            loss = model.forward(input_sample)
            loss.backward()
            epoch_loss += loss.item() * length
            # print("sample\t" + str(j) + "\tloss\t" + str(loss.item()))

            if ((j + 1) % batch) == 0 or (j == len(train_dataset) - 1):
                # print("updated")
                optimizer.step()
                optimizer.zero_grad()
        epoch_loss = epoch_loss / (total_length-1)
        time.sleep(0.5)
        print('')
        logger.info("Epoch:\t" + str(i) + "\t Training loss:\t" + str(round(epoch_loss, 4)) + "\t PPL: " + str(round(np.exp(epoch_loss), 4)))
        epoch_loss_list.append(epoch_loss)
        # evaluate
        dev_loss = evaluate(dev_dataset, model, device)
        test_loss = evaluate(test_dataset, model, device)
        time.sleep(0.5)
        print('')
        logger.info("\t\t Dev Loss: " + str(round(dev_loss, 4)) + "\t PPL: " + str(round(np.exp(dev_loss), 4)))
        logger.info("\t\t Test Loss: " + str(round(test_loss, 4)) + "\t PPL: " + str(round(np.exp(test_loss), 4)))
    # print(epoch_loss_list)


if __name__ == '__main__':
    main()
