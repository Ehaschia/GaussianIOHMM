# coding: utf-8
import argparse
import json
import math
import random
from typing import List

from torch.optim import SGD
from torch.optim.adamw import AdamW

from tqdm import tqdm

from io_module.data_loader import *
from io_module.logger import *
from model.sequence_labeling import *
from model.weighted_iohmm import WeightIOHMM
from optim.lr_scheduler import ExponentialScheduler

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

def get_optimizer(parameters, optim, learning_rate, amsgrad, weight_decay, lr_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=amsgrad,
                          weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler

def evaluate(data, batch, model, device):
    model.eval()
    total_token_num = 0
    corr_token_num = 0
    with torch.no_grad():
        for i in range(math.ceil(len(data) / batch)):
            sentences, labels, masks, revert_order = standardize_batch(data[i * batch: (i + 1) * batch])
            acc, corr = model.get_acc(sentences.squeeze().to(device),
                                      labels.squeeze().to(device),
                                      masks.squeeze().to(device))
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
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    parser.add_argument('--amsgrad', action='store_true', help='AMD Grad')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--warmup_steps', type=int, default=0, metavar='N', help='number of steps to warm up (default: 0)')
    parser.add_argument('--var_scale', type=float, default=1.0)
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument('--in_mu_drop', type=float, default=0.0)
    parser.add_argument('--in_cho_drop', type=float, default=0.0)
    parser.add_argument('--t_mu_drop', type=float, default=0.0)
    parser.add_argument('--t_cho_drop', type=float, default=0.0)
    parser.add_argument('--out_mu_drop', type=float, default=0.0)
    parser.add_argument('--out_cho_drop', type=float, default=0.0)
    parser.add_argument('--trans_cho_method', type=str, choices=['random', 'wishart'], default='random')
    parser.add_argument('--input_cho_init', type=float, default=1.0,
                        help='init method of input cholesky matrix. 0 means random. The other score means constant')
    parser.add_argument('--trans_cho_init', type=float, default=1.0,
                        help='init added scale of random version init_cho_init')
    parser.add_argument('--output_cho_init', type=float, default=1.0,
                        help='init method of output cholesky matrix. 0 means random. The other score means constant')
    # i_comp_num = 1, t_comp_num = 1, o_comp_num = 1, max_comp = 1,
    parser.add_argument('--input_comp_num', type=int, default=1, help='input mixture gaussian component number')
    parser.add_argument('--tran_comp_num', type=int, default=2, help='transition mixture gaussian component number')
    parser.add_argument('--output_comp_num', type=int, default=1, help='output mixture gaussian component number')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='pruning hyper-parameter, greater than 1 is max component, less than 1 is max value')
    parser.add_argument('--tran_weight', type=float, default=0.0001)
    parser.add_argument('--input_weight', type=float, default=0.0)
    parser.add_argument('--output_weight', type=float, default=0.0)
    parser.add_argument('--emission_cho_grad', type=bool, default=False)
    parser.add_argument('--transition_cho_grad', type=bool, default=True)
    parser.add_argument('--decode_cho_grad', type=bool, default=False)
    parser.add_argument('--gaussian_decode', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    log_dir = args.log_dir
    batch_size = args.batch

    # setting optimizer
    batch_size = args.batch
    optim = args.optim
    lr = args.lr
    lr_decay = args.lr_decay
    warmup_steps = args.warmup_steps
    amsgrad = args.amsgrad
    weight_decay = args.weight_decay

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
    threshold = args.threshold
    normalize_weight = [args.tran_weight, args.input_weight, args.output_weight]
    gaussian_decode = args.gaussian_decode

    EMISSION_CHO_GRAD = args.emission_cho_grad
    TRANSITION_CHO_GRAD = args.transition_cho_grad
    DECODE_CHO_GRAD = args.decode_cho_grad

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

    device = torch.device('cuda') if args.gpu else torch.device('cpu')
    # Loading data
    logger.info('Load data....')
    train_dataset = read_sequence_labeling_data(root, type='train')
    dev_dataset = read_sequence_labeling_data(root, type='dev')
    test_dataset = read_sequence_labeling_data(root, type='test')

    # build model
    if threshold >= 1.0:
        model = MixtureGaussianSequenceLabeling(dim=args.dim, ntokens=ntokens, nlabels=nlabels,
                                                t_cho_method=tran_cho_method, t_cho_init=trans_cho_init,
                                                in_cho_init=input_cho_init, out_cho_init=output_cho_init,
                                                in_mu_drop=in_mu_drop, in_cho_drop=in_cho_drop,
                                                t_mu_drop=t_mu_drop, t_cho_drop=t_cho_drop,
                                                out_mu_drop=out_mu_drop, out_cho_drop=out_cho_drop,
                                                i_comp_num=input_num_comp, t_comp_num=tran_num_comp,
                                                o_comp_num=output_num_comp, max_comp=int(threshold),
                                                gaussian_decode=gaussian_decode)
    else:
        model = ThresholdPruningMGSL(dim=args.dim, ntokens=ntokens, nlabels=nlabels,
                                     t_cho_method=tran_cho_method, t_cho_init=trans_cho_init,
                                     in_cho_init=input_cho_init, out_cho_init=output_cho_init,
                                     in_mu_drop=in_mu_drop, in_cho_drop=in_cho_drop,
                                     t_mu_drop=t_mu_drop, t_cho_drop=t_cho_drop,
                                     out_mu_drop=out_mu_drop, out_cho_drop=out_cho_drop,
                                     i_comp_num=input_num_comp, t_comp_num=tran_num_comp,
                                     o_comp_num=output_num_comp, threshold=threshold,
                                     gaussian_decode=gaussian_decode)

    # model = RNNSequenceLabeling("RNN_TANH", ntokens=ntokens, nlabels=nlabels, ninp=10, nhid=10)
    # model = WeightIOHMM(vocab_size=ntokens, nlabel=nlabels, num_state=args.dim)
    model.to(device)
    logger.info('Building model ' + model.__class__.__name__ + '...')
    parameters_need_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, scheduler = get_optimizer(parameters_need_update, optim, lr, amsgrad, weight_decay,
                                         lr_decay=lr_decay, warmup_steps=warmup_steps)
    # depend on dev ppl
    best_epoch = (-1, 0.0, 0.0)

    # Default: util 6 epoch not update best_epoch
    # If aim_epoch is not 0. It will train aim_epoch times.
    def train(best_epoch, thread=6, aim_epoch=0):
        epoch = best_epoch[0] + 1
        while epoch - best_epoch[0] <= thread:
            epoch_loss = 0
            random.shuffle(train_dataset)
            model.train()
            optimizer.zero_grad()
            for j in tqdm(range(math.ceil(len(train_dataset) / batch_size))):
                samples = train_dataset[j * batch_size: (j + 1) * batch_size]

                sentences, labels, masks, revert_order = standardize_batch(samples)
                loss = 0
                if threshold >= 1.0:
                    loss = model.get_loss(sentences, labels, masks, normalize_weight=normalize_weight)
                else:
                    for i in range(batch_size):
                        loss += model.get_loss(sentences[i], labels[i], masks[i], normalize_weight=normalize_weight)
                # loss = model.get_loss(sentences.to(device), labels.to(device), masks.to(device))
                # loss = model.get_loss(sentences, labels, masks, normalize_weight=normalize_weight)
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += (loss.item()) # * sentences.size(0)
            logger.info('Epoch ' + str(epoch) + ' Loss: ' + str(round(epoch_loss / len(train_dataset), 4)))
            if threshold >= 1.0:
                acc, corr = evaluate(dev_dataset, batch_size, model, device)
            else:
                acc, corr = evaluate(dev_dataset, 1, model, device)
            logger.info('\t Dev Acc: ' + str(round(acc * 100, 3)))
            if best_epoch[1] < acc:
                if threshold >= 1.0:
                    test_acc, _ = evaluate(test_dataset, batch_size, model, device)
                else:
                    test_acc, _ = evaluate(test_dataset, 1, model, device)
                logger.info('\t Test Acc: ' + str(round(test_acc * 100, 3)))
                best_epoch = (epoch, acc, test_acc)
            epoch += 1

            if aim_epoch != 0 and epoch >= aim_epoch:
                break

        logger.info("Best Epoch: " + str(best_epoch[0]) + " Dev ACC: " + str(round(best_epoch[1] * 100, 3)) +
                    "Test ACC: " + str(round(best_epoch[2] * 100, 3)))
        return best_epoch

    # for parameter in model.parameters():
    #     # flip
    #     parameter.requires_grad = not parameter.requires_grad

    best_epoch = train(best_epoch, thread=50)

    # logger.info("After tunning var. Here we tunning mu")
    #
    # for parameter in model.parameters():
    #     # flip
    #     parameter.requires_grad = not parameter.requires_grad
    #
    # best_epoch = train(best_epoch)

    with open(log_dir + '/' + 'result.json', 'w') as f:
        final_result = {"Epoch": best_epoch[0],
                        "Dev": best_epoch[1] * 100,
                        "Test": best_epoch[2] * 100}
        json.dump(final_result, f)


if __name__ == '__main__':
    main()
