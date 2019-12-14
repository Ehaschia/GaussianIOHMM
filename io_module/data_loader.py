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


def sequence_labeling_data_loader(root, type='train'):
    name = {'train': 'synthetic.train',
            'dev': 'synthetic.dev',
            'test': 'synthetic.test'}
    datas = []
    with open(root + '/' + name[type], 'r') as f:
        sentence = []
        labels = []
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                datas.append((sentence, labels))
                sentence = []
                labels = []
            else:
                str_word, str_label = line.split('\t')
                sentence.append(int(str_word))
                labels.append(int(str_label))
    return datas
