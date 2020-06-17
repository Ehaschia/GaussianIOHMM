# coding: utf-8
import argparse
import json
import random

from torch.optim import SGD
from torch.optim.adamw import AdamW

from io_module import conllx_data
from io_module.logger import *
from io_module.utils import iterate_data
from model.LM import *
# from misc.sharp_detect import SharpDetector
from optim.lr_scheduler import ExponentialScheduler


def evaluate(data, batch, model, device):
    model.eval()

    total_ppl = 0
    word_cnt = 0
    with torch.no_grad():
        for batch_data in iterate_data(data, batch):
            # sentences, labels, masks, revert_order = standardize_batch(data[i * batch: (i + 1) * batch])
            words = batch_data['WORD'].to(device)
            masks = batch_data['MASK'].to(device)
            lengths = batch_data['LENGTH']
            ppl = model.get_loss(words, masks)
            total_ppl += ppl.item() * words.size(0)
            word_cnt += torch.sum(lengths).item()
    return total_ppl / word_cnt


def get_optimizer(parameters, optim, learning_rate, amsgrad, weight_decay, lr_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, amsgrad=amsgrad,
                          weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler


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
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    parser.add_argument('--amsgrad', action='store_true', help='AMD Grad')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight for l2 norm decay')
    parser.add_argument('--warmup_steps', type=int, default=0, metavar='N',
                        help='number of steps to warm up (default: 0)')
    parser.add_argument('--log_dir', type=str,
                        default='./output/' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + "/")
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument('--unk_replace', type=float, default=0.0, help='The rate to replace a singleton word with UNK')
    parser.add_argument('--model', choices=['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU', 'MultiplicativeRNN'], default='RNN_TANH')
    parser.add_argument('--active', choices=['sigmoid', 'tanh', 'relu'], default='sigmoid')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--symbolic_start', type=bool, default=False)
    parser.add_argument('--symbolic_end', type=bool, default=False)

    args = parser.parse_args()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    log_dir = args.log_dir

    # setting optimizer
    optim = args.optim
    lr = args.lr
    lr_decay = args.lr_decay
    warmup_steps = args.warmup_steps
    amsgrad = args.amsgrad
    weight_decay = args.weight_decay

    # data
    root = args.data
    unk_replace = args.unk_replace
    s_start = args.symbolic_start
    s_end = args.symbolic_end

    # model
    model_type = args.model
    dim = args.dim
    batch_size = args.batch
    active_func = args.active
    dropout = args.dropout

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # save parameter
    save_parameter_to_json(log_dir, vars(args))

    logger = get_logger('LanguageModel')
    change_handler(logger, log_dir)
    # logger = LOGGER
    logger.info(args)

    device = torch.device('cuda') # if args.gpu else torch.device('cpu')

    # Loading data
    logger.info('Load PTB data....')
    alphabet_path = os.path.join(root, 'alphabets')
    train_path = os.path.join(root, 'train.conllu')
    dev_path = os.path.join(root, 'dev.conllu')
    test_path = os.path.join(root, 'test.conllu')
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path, test_path],
                                                                                             embedd_dict=None,
                                                                                             max_vocabulary_size=1e4,
                                                                                             min_occurrence=1,
                                                                                             unk_rank=0)

    train_dataset = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                   symbolic_root=s_start, symbolic_end=s_end)
    num_data = sum(train_dataset[1])
    dev_dataset = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                        symbolic_root=s_start, symbolic_end=s_end)
    test_dataset = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                         symbolic_root=s_start, symbolic_end=s_end)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    ntokens = word_alphabet.size()

    if model_type in ['LSTM', 'RNN_TANH', 'RNN_RELU', 'GRU']:
        model = RNNLanguageModel(model_type, ntokens=ntokens, ninp=dim, nhid=dim, dropout=dropout)
    elif model_type == 'MultiplicativeRNN':
        model = MultiplicativeRNN(active_func, ntokens=ntokens, ninp=dim, nhid=dim, dropout=dropout)
    else:
        raise ValueError("Error model type")

    model.to(device)
    logger.info('Building model ' + model.__class__.__name__ + '...')
    parameters_need_update = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, scheduler = get_optimizer(parameters_need_update, optim, lr, amsgrad, weight_decay,
                                         lr_decay=lr_decay, warmup_steps=warmup_steps)
    # depend on dev ppl
    best_epoch = (-1, 1e8, 0.0)
    num_batches = num_data // batch_size + 1

    def train(best_epoch, thread=6):
        epoch = 0
        while epoch - best_epoch[0] <= thread:
            epoch_loss = 0
            num_back = 0
            num_words = 0
            num_insts = 0
            model.train()
            for step, data in enumerate(iterate_data(train_dataset, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
                # for j in tqdm(range(math.ceil(len(train_dataset) / batch_size))):
                optimizer.zero_grad()
                # samples = train_dataset[j * batch_size: (j + 1) * batch_size]
                words, masks = data['WORD'].to(device), data['MASK'].to(device)

                loss = model.get_loss(words, masks)
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item() * words.size(0)
                num_words += torch.sum(masks).item()
                num_insts += words.size()[0]

                if step % 10 == 0:
                    torch.cuda.empty_cache()
                    sys.stdout.write("\b" * num_back)
                    sys.stdout.write(" " * num_back)
                    sys.stdout.write("\b" * num_back)
                    curr_lr = scheduler.get_lr()[0]
                    log_info = '[%d/%d (%.0f%%) lr=%.6f] loss: %.4f (%.4f)' % (
                        step, num_batches, 100. * step / num_batches,
                        curr_lr, epoch_loss / num_insts, epoch_loss / num_words)
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)
            logger.info('Epoch ' + str(epoch) + ' Loss: ' + str(round(epoch_loss / num_insts, 4)))

            ppl = evaluate(dev_dataset, batch_size, model, device)

            logger.info('\t Dev PPL: ' + str(round(ppl, 3)))

            if best_epoch[1] > ppl:
                test_ppl = evaluate(test_dataset, batch_size, model, device)
                logger.info('\t Test PPL: ' + str(round(test_ppl, 3)))
                best_epoch = (epoch, ppl, test_ppl)
                patient = 0
            else:
                patient += 1
            epoch += 1
            if patient > 4:
                print('reset optimizer momentums')
                scheduler.reset_state()
                patient = 0

        logger.info("Best Epoch: " + str(best_epoch[0]) + " Dev ACC: " + str(round(best_epoch[1], 3)) +
                    "Test ACC: " + str(round(best_epoch[2], 3)))
        return best_epoch

    best_epoch = train(best_epoch, thread=10)

    with open(log_dir + '/' + 'result.json', 'w') as f:
        final_result = {"Epoch": best_epoch[0],
                        "Dev": best_epoch[1],
                        "Test": best_epoch[2]}
        json.dump(final_result, f)


if __name__ == '__main__':
    main()
