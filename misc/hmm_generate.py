import numpy as np

# generate data by hmm

ntokens = 10

train_size = 1000
dev_size = 100
test_size = 100

np.random.seed(10)

trans_matrix = np.random.rand(ntokens + 1, ntokens + 1)


def normalize(vector):
    nsum = np.sum(vector)
    return vector / nsum


for i in range(ntokens):
    trans_matrix[i] = normalize(trans_matrix[i])


def get_next(prob):
    idx = np.random.choice(len(prob), 1, p=prob)
    return int(idx[0])


def generate_dataset(size, matrix):
    dataset = []
    while len(dataset) < size:
        idx = int(np.random.choice(len(matrix[0]), 1)[0])
        sentence = []
        while idx != ntokens:
            sentence.append(idx)
            idx = get_next(matrix[idx])
        if len(sentence) > 1:
            dataset.append(sentence)
    return dataset


def save_dataset(root, name, dataset):
    with open(root + '/' + name, 'w') as f:
        for sample in dataset:
            str_sample = ' '.join(str(i) for i in sample)
            str_sample = str_sample.strip()
            f.write(str_sample)
            f.write('\n')


root = 'E:/Code/GaussianIOHMM/dataset/hmm_generate/'
train_dataset = generate_dataset(train_size, trans_matrix)
save_dataset(root, 'train.txt', train_dataset)
dev_dataset = generate_dataset(dev_size, trans_matrix)
save_dataset(root, 'valid.txt', dev_dataset)
test_dataset = generate_dataset(test_size, trans_matrix)
save_dataset(root, 'test.txt', test_dataset)
