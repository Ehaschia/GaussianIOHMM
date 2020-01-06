# coding: utf-8
import argparse
import json
import random

from torch.optim import SGD
from torch.optim.adamw import AdamW

from io_module import conllx_data
from io_module.logger import *
from io_module.utils import iterate_data
from model.sequence_labeling import *
from model.weighted_iohmm import WeightIOHMM


def evaluate(data, batch, model, device):
    model.eval()
    total_token_num = 0
    corr_token_num = 0
    with torch.no_grad():
        for batch_data in iterate_data(data, batch):
            # sentences, labels, masks, revert_order = standardize_batch(data[i * batch: (i + 1) * batch])
            words = batch_data['WORD'].to(device)
            labels = batch_data['POS'].to(device)
            masks = batch_data['MASK'].to(device)
            acc, corr = model.get_acc(words, labels, masks)
            corr_token_num += corr
            total_token_num += torch.sum(masks).item()
    return corr_token_num / total_token_num, corr_token_num


# TODO(lwzhang) add ExponentialScheduler or not
# https://github.com/XuezheMax/NeuroNLP2/blob/328ab3a15876257c69fa1f96cb522e45c7d21c6d/experiments/pos_tagging.py#L28
def get_optimizer(parameters, optim, learning_rate, amsgrad, weight_decay):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=amsgrad,
                          weight_decay=weight_decay)
    # init_lr = 1e-7
    # scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer  # , scheduler


def save_parameter_to_json(path, parameters):
    with open(path + 'param.json', 'w') as f:
        json.dump(parameters, f)


def main():
    parser = argparse.ArgumentParser(description="Gaussian Input Output HMM")

    parser.add_argument(
        '--data',
        type=str,
        default='./dataset/ptb/',
        help='location of the data corpus')
    parser.add_argument('--batch', type=int, default=256)
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
    parser.add_argument('--unk_replace', type=float, default=0.0, help='The rate to replace a singleton word with UNK')
    parser.add_argument('--optim', choices=['sgd', 'adam'])
    parser.add_argument('--amsgrad', action='store_true', help='AMD Grad')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    parser.add_argument('--tran_weight', type=float, default=0.0001)
    parser.add_argument('--input_weight', type=float, default=0.0)
    parser.add_argument('--output_weight', type=float, default=0.0)
    parser.add_argument('--emission_cho_grad', type=bool, default=False)
    parser.add_argument('--transition_cho_grad', type=bool, default=True)
    parser.add_argument('--decode_cho_grad', type=bool, default=False)

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
    unk_replace = args.unk_replace
    optim = args.optim
    amsgrad = args.amsgrad
    weight_decay = args.weight_decay
    normalize_weight = [args.tran_weight, args.input_weight, args.output_weight]

    EMISSION_CHO_GRAD = args.emission_cho_grad
    TRANSITION_CHO_GRAD = args.transition_cho_grad
    DECODE_CHO_GRAD = args.decode_cho_grad

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # save parameter
    save_parameter_to_json(log_dir, vars(args))

    logger = get_logger('Sequence-Labeling')
    change_handler(logger, log_dir)
    # logger = LOGGER
    logger.info(args)

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    # Loading data
    logger.info('Load PTB data....')
    alphabet_path = os.path.join(root, 'alphabets')
    train_path = os.path.join(root, 'train.conllu')
    dev_path = os.path.join(root, 'dev.conllu')
    test_path = os.path.join(root, 'test.conllu')
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path,
                                                                                             train_path,
                                                                                             data_paths=[dev_path,
                                                                                                         test_path],
                                                                                             embedd_dict=None,
                                                                                             max_vocabulary_size=1e5)
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    ntokens = word_alphabet.size()
    nlabels = pos_alphabet.size()

    train_dataset = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                   type_alphabet)
    num_data = sum(train_dataset[1])
    dev_dataset = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    test_dataset = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # build model
    model = MixtureGaussianSequenceLabeling(dim=args.dim, ntokens=ntokens, nlabels=nlabels,
                                            t_cho_method=tran_cho_method, t_cho_init=trans_cho_init,
                                            in_cho_init=input_cho_init, out_cho_init=output_cho_init,
                                            in_mu_drop=in_mu_drop, in_cho_drop=in_cho_drop,
                                            t_mu_drop=t_mu_drop, t_cho_drop=t_cho_drop,
                                            out_mu_drop=out_mu_drop, out_cho_drop=out_cho_drop,
                                            i_comp_num=input_num_comp, t_comp_num=tran_num_comp,
                                            o_comp_num=output_num_comp, max_comp=max_comp)

    # model = RNNSequenceLabeling("LSTM", ntokens=ntokens, nlabels=nlabels, ninp=100, nhid=100)
    # model = WeightIOHMM(vocab_size=ntokens, nlabel=nlabels, num_state=10)
    model.to(device)
    logger.info('Building model ' + model.__class__.__name__ + '...')
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    parameters_need_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_optimizer(parameters_need_update, optim, lr, amsgrad, weight_decay)
    # depend on dev ppl
    best_epoch = (-1, 0.0, 0.0)
    # util 6 epoch not update best_epoch
    num_batches = num_data // batch_size + 1

    def train(best_epoch, thread=6):
        epoch = 0
        while epoch - best_epoch[0] <= thread:
            epoch_loss = 0
            num_back = 0
            num_words = 0
            num_insts = 0
            model.train()
            for step, data in enumerate(
                    iterate_data(train_dataset, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
                # for j in tqdm(range(math.ceil(len(train_dataset) / batch_size))):
                optimizer.zero_grad()
                # samples = train_dataset[j * batch_size: (j + 1) * batch_size]
                words, labels, masks = data['WORD'].to(device), data['POS'].to(device), data['MASK'].to(device)

                # sentences, labels, masks, revert_order = standardize_batch(samples)
                # loss = model.get_loss(words, labels, masks, normalize_weight=normalize_weight)
                loss = model.get_loss(words, labels, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += (loss.item()) * words.size(0)
                num_words += torch.sum(masks).item()
                num_insts += words.size()[0]
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    log_info = '[%d/%d (%.0f%%) lr=%.6f] loss: %.4f (%.4f)' % (
                    step, num_batches, 100. * step / num_batches,
                    lr, epoch_loss / num_insts, epoch_loss / num_words)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)

            logger.info('Epoch ' + str(epoch) + ' Loss: ' + str(round(epoch_loss / num_insts, 4)))
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

    best_epoch = train(best_epoch, thread=6)
    # logger.info("After tunning mu. Here we tunning variance")
    # # flip gradient
    #
    # for parameter in model.parameters():
    #     # flip
    #     parameter.requires_grad = not parameter.requires_grad
    
    # best_epoch = train(best_epoch)
    with open(log_dir + '/' + 'result.json', 'w') as f:
        final_result = {"Epoch": best_epoch[0],
                        "Dev": best_epoch[1] * 100,
                        "Test": best_epoch[2] * 100}
        json.dump(final_result, f)


if __name__ == '__main__':
    main()
