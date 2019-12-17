# coding: utf-8
import argparse
import json
import math
import random
from typing import List

import torch.optim as optim
from tqdm import tqdm

from io_module.data_loader import *
from io_module.logger import *
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
    return torch.tensor(standardized_sentence_list).long(), torch.tensor(standardized_label_list).long(), \
           torch.tensor(mask_list).long(), torch.tensor(revert_idx_list).long()


def evaluate(data, batch, model, device):
    model.eval()
    total_token_num = 0
    corr_token_num = 0
    with torch.no_grad():
        for i in range(math.ceil(len(data) / batch)):
            sentences, labels, masks, revert_order = standardize_batch(data[i * batch: (i + 1) * batch])
            acc, corr = model.get_acc(sentences.to(device), labels.to(device), masks.to(device))
            corr_token_num += corr
            total_token_num += torch.sum(masks).item()
    return corr_token_num / total_token_num, corr_token_num


def save_parameter_to_json(path, parameters):
    with open(path + 'param.json', 'w') as f:
        json.dump(parameters, f)


def main():
    parser = argparse.ArgumentParser(description="Gaussian Input Output HMM")

    parser.add_argument(
        '--data',
        type=str,
        default='./dataset/syntic_data_yong/0-1000-10-new',
        help='location of the data corpus')
    parser.add_argument('--batch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--var_scale', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument('--in_mu_drop', type=float, default=0.0)
    parser.add_argument('--in_cho_drop', type=float, default=0.0)
    parser.add_argument('--t_mu_drop', type=float, default=0.0)
    parser.add_argument('--t_cho_drop', type=float, default=0.0)
    parser.add_argument('--out_mu_drop', type=float, default=0.0)
    parser.add_argument('--out_cho_drop', type=float, default=0.0)
    parser.add_argument('--trans_cho_method', type=str, choices=['random', 'wishart'], default='random')
    parser.add_argument('--input_cho_init', type=float, default=0.0,
                        help='init method of input cholesky matrix. 0 means random. The other score means constant')
    parser.add_argument('--trans_cho_init', type=float, default=1.0,
                        help='init added scale of random version init_cho_init')
    parser.add_argument('--output_cho_init', type=float, default=0.0,
                        help='init method of output cholesky matrix. 0 means random. The other score means constant')
    # i_comp_num = 1, t_comp_num = 1, o_comp_num = 1, max_comp = 1,
    parser.add_argument('--input_comp_num', type=int, default=1,
                        help='input mixture gaussian component number')
    parser.add_argument('--tran_comp_num', type=int, default=1,
                        help='transition mixture gaussian component number')
    parser.add_argument('--output_comp_num', type=int, default=1,
                        help='output mixture gaussian component number')
    parser.add_argument('--max_comp', type=int, default=1,
                        help='number of max number of component')

    args = parser.parse_args()
    # np.random.seed(global_variables.RANDOM_SEED)
    # torch.manual_seed(global_variables.RANDOM_SEED)
    # random.seed(global_variables.RANDOM_SEED)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    log_dir = args.log_dir
    batch_size = args.batch
    lr = args.lr
    momentum = args.momentum
    root = args.data
    in_mu_drop = args.in_mu_drop
    in_cho_drop = args.in_cho_drop
    t_mu_drop = args.t_mu_drop
    t_cho_drop = args.t_cho_drop
    out_mu_drop = args.out_mu_drop
    out_cho_drop = args.out_cho_drop
    tran_cho_method = args.trans_cho_method
    input_cho_init = args.input_cho_init
    trans_cho_init = args.trans_cho_init
    output_cho_init = args.output_cho_init
    input_num_comp = args.input_comp_num
    tran_num_comp = args.tran_comp_num
    output_num_comp = args.output_comp_num
    max_comp = args.max_comp

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_parameter_to_json(log_dir, vars(args))
    # TODO ntokens generate from dataset
    ntokens = 1000
    nlabels = 5
    # save parameter
    logger = get_logger('Sequence-Labeling')
    change_handler(logger, log_dir)
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
    # Loading data
    logger.info('Load data....')
    train_dataset = sequence_labeling_data_loader(root, type='train')
    dev_dataset = sequence_labeling_data_loader(root, type='dev')
    test_dataset = sequence_labeling_data_loader(root, type='test')

    # build model
    model = MixtureGaussianSequenceLabeling(dim=args.dim, ntokens=ntokens, nlabels=nlabels,
                                            t_cho_method=tran_cho_method, t_cho_init=trans_cho_init,
                                            in_cho_init=input_cho_init, out_cho_init=output_cho_init,
                                            in_mu_drop=in_mu_drop, in_cho_drop=in_cho_drop,
                                            t_mu_drop=t_mu_drop, t_cho_drop=t_cho_drop,
                                            out_mu_drop=out_mu_drop, out_cho_drop=out_cho_drop,
                                            i_comp_num=input_num_comp, t_comp_num=tran_num_comp,
                                            o_comp_num=output_num_comp, max_comp=max_comp)

    # model = RNNSequenceLabeling("RNN_TANH", ntokens=ntokens, nlabels=nlabels, ninp=10, nhid=10)
    model.to(device)
    logger.info('Building model ' + model.__class__.__name__ + '...')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # depend on dev ppl
    best_epoch = (-1, 0.0, 0.0)

    # util 6 epoch not update best_epoch
    def train(best_epoch, thread=6):
        epoch = 0
        while epoch - best_epoch[0] <= thread:
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
            logger.info('Epoch ' + str(epoch) + ' Loss: ' + str(round(epoch_loss / len(train_dataset), 4)))
            acc, corr = evaluate(dev_dataset, batch_size, model, device)
            logger.info('\t Dev Acc: ' + str(round(acc * 100, 3)))
            if best_epoch[1] < acc:
                test_acc, _ = evaluate(test_dataset, batch_size, model, device)
                logger.info('\t Test Acc: ' + str(round(test_acc * 100, 3)))
                best_epoch = (epoch, acc, test_acc)
            epoch += 1

        logger.info("Best Epoch: " + str(best_epoch[0]) + " Dev ACC: " + str(round(best_epoch[1] * 100, 3)) +
                    "Test ACC: " + str(round(best_epoch[2] * 100, 3)))
        return best_epoch

    # for parameter in model.parameters():
    #     # flip
    #     parameter.requires_grad = not parameter.requires_grad

    best_epoch = train(best_epoch, thread=6)

    logger.info("After tunning var. Here we tunning mu")

    for parameter in model.parameters():
        # flip
        parameter.requires_grad = not parameter.requires_grad

    best_epoch = train(best_epoch)


    with open(log_dir + '/' + 'result.json', 'w') as f:
        final_result = {"Epoch": best_epoch[0],
                        "Dev": best_epoch[1] * 100,
                        "Test": best_epoch[2] * 100}
        json.dump(final_result, f)


if __name__ == '__main__':
    main()
