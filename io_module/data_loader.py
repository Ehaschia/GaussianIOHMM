

# used for load sentence in a line
def hmm_generate_data_loader(root, type='train'):
    name = {'train': 'train.txt',
            'dev': 'valid.txt',
            'test': 'test.txt'}
    datas = []
    with open(root + '/' + name[type], 'r') as f:
        for line in f.readlines():
            sentence = []
            line = line.split(' ')
            for i in line:
                sentence.append(int(i))
            datas.append(sentence)
    return datas
