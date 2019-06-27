# tmp for run

def syntic_data(root, type='train'):
    dir = '/syntic data_yong/synth/0-1000-10-new/'
    name = {'train': 'synthetic.train',
            'dev': 'synthetic.dev',
            'test': 'synthetic.test'}
    datas = []
    with open(root + dir + name[type], 'r') as f:
        sentence = []
        line = f.readline()
        while line:
            if len(line.strip()) == 0:
                datas.append(sentence)
                sentence = []
            else:
                word = int(line.split('\t')[0])
                sentence.append(word)
            line = f.readline()
    return datas


# used for load sentence in a line
def data_loader(root, type='train'):
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
