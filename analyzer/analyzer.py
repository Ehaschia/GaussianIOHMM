class Analyzer:
    def __init__(self, word_alphabet, pos_alphabet):
        self.__word_alphabet = word_alphabet
        self.__pos_alphabet = pos_alphabet

    # error of every pos tag and every word
    # error of every position
    def error_rate(self, sents, preds, goldens, length, path):

        right_pos = {}
        error_pos = {}
        re_pos_mat = {}

        total_word = {}
        error_word = {}
        re_word_mat = {}

        for idx, (pred, golden, sent) in enumerate(zip(preds, goldens, sents)):
            for j in range(length[idx]):
                if sent[j] not in total_word:
                    total_word[sent[j]] = 0
                total_word[sent[j]] += 1

                if pred[j] == golden[j]:
                    if golden[j] not in right_pos:
                        right_pos[golden[j]] = 0
                    right_pos[golden[j]] += 1
                else:
                    if golden[j] not in error_pos:
                        error_pos[golden[j]] = 0
                    error_pos[golden[j]] += 1

                    if sent[j] not in error_word:
                        error_word[sent[j]] = 0
                    error_word[sent[j]] += 1

                    # consider right-error distribution
                    if golden[j] not in re_pos_mat:
                        re_pos_mat[golden[j]] = {}
                    if pred[j] not in re_pos_mat[golden[j]]:
                        re_pos_mat[golden[j]][pred[j]] = 0
                    re_pos_mat[golden[j]][pred[j]] += 1

                    # transition in word:
                    if sent[j] not in re_word_mat:
                        re_word_mat[sent[j]] = {}
                    if golden[j] not in re_word_mat[sent[j]]:
                        re_word_mat[sent[j]][golden[j]] = {}
                    if pred[j] not in re_word_mat[sent[j]][golden[j]]:
                        re_word_mat[sent[j]][golden[j]][pred[j]] = 0
                    re_word_mat[sent[j]][golden[j]][pred[j]] += 1

        #save
        with open(path + '_pos_right.txt', 'w') as f:
            for item in right_pos.items():
                f.write(self.__pos_alphabet.instances[item[0]] + '\t' + str(item[1]) + '\n')

        with open(path + '_pos_error.txt', 'w') as f:
            for item in error_pos.items():
                f.write(self.__pos_alphabet.instances[item[0]] + '\t' + str(item[1]) + '\n')

        with open(path + '_pos_rematrix.txt', 'w') as f:
            for item in re_pos_mat.items():
                for item1 in item[1].items():
                    f.write(self.__pos_alphabet.instances[item[0]] + '~' + self.__pos_alphabet.instances[item1[0]] + '\t' + str(item1[1]) + '\n')

        with open(path + '_word_right.txt', 'w') as f:
            for item in right_pos.items():
                f.write(self.__word_alphabet.instances[item[0]] + '\t' + str(item[1]) + '\n')

        with open(path + '_word_error.txt', 'w') as f:
            for item in error_pos.items():
                f.write(self.__word_alphabet.instances[item[0]] + '\t' + str(item[1]) + '\n')

        with open(path + '_word_rematrix.txt', 'w') as f:
            for word in re_word_mat.items():
                for golden in word[1].items():
                    for pred in golden[1].items():
                        f.write(self.__word_alphabet.instances[word[0]] + '~' +
                                self.__pos_alphabet.instances[golden[0]] + '~' +
                                self.__pos_alphabet.instances[pred[0]] + '\t' + str(item1[1]) + '\n')
