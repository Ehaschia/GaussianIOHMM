from io_module import conllx_data
import os
import numpy as np

# load alphabet and get unk and it's pos tag
root = 'dataset/ptb/'
alphabet_path = os.path.join(root, 'alphabets')
train_path = os.path.join(root, 'train.conllu')
dev_path = os.path.join(root, 'dev.conllu')
test_path = os.path.join(root, 'test.conllu')

word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                         data_paths=[dev_path, test_path],
                                                                                         embedd_dict=None,
                                                                                         max_vocabulary_size=1e5,
                                                                                         min_occurrence=5)
# load dataset
train_dataset = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
dev_dataset = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
test_dataset = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

def unk_distributon(train_dataset):
    train_word, train_char, train_pos, train_single, train_lens, train_mask = \
        train_dataset[0]['WORD'].numpy(), train_dataset[0]['CHAR'].numpy(), \
        train_dataset[0]['POS'].numpy(), train_dataset[0]['SINGLE'].numpy(), \
        train_dataset[0]['LENGTH'].numpy(), train_dataset[0]['MASK'].numpy()

    single_rate = np.sum(train_single) / np.sum(train_mask)
    print("Train Single Rate: \t" + str(single_rate))
    sig_dis = {}
    for i in range(len(train_single)):
        for j in range(train_lens[i]):
            if train_single[i][j]:
                if train_pos[i][j] not in sig_dis:
                    sig_dis[train_pos[i][j]] = []
                sig_dis[train_pos[i][j]].append(train_word[i][j])

    # present current distribution:
    print("-"*10 + 'UNK distribution' + '-'*10)
    for item in sig_dis.items():
        print(pos_alphabet.instances[item[0]] + ':\t' + str(len(item[1])))

print('\n\n-----Train Dataset----\n\n')
unk_distributon(train_dataset)
print('\n\n-----Dev Dataset----\n\n')
unk_distributon(dev_dataset)
print('\n\n-----Test Dataset----\n\n')
unk_distributon(test_dataset)

