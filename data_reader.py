from __future__ import print_function
from __future__ import division

import os
import codecs
import collections
import numpy as np
import pickle

class Vocab:

    def __init__(self, token2index=None, index2token=None, max_vocab_size=None):
        self._token2index = token2index or {}
        self._index2token = index2token or []
        self._vocab_size = max_vocab_size

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    @property
    def token2index(self):
        return self._token2index

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=0):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)

        return cls(token2index, index2token)


def load_data(data_dir, max_doc_length=10, max_sent_length=50, max_vocab_size=150000, data_set=['train','test']):

    try:
        word_vocab = Vocab.load(filename='vocab')
    except Exception as e:
        print("build vocaburary")
        word_vocab = Vocab(max_vocab_size=max_vocab_size)
        word_vocab.feed('<unk>')
        word_vocab.feed('{')
        word_vocab.feed('}')

    actual_max_doc_length = 0

    word_tokens = collections.defaultdict(list)
    labels = collections.defaultdict(list)
    questions = collections.defaultdict(list)
    sentences = collections.defaultdict(list)

    for fname in data_set:
        print('reading', fname)
        pname = os.path.join(data_dir, fname)
        fn = len(os.listdir(pname))
        dnames = ['%s.txt' % i for i in range(fn)]
        for dname in dnames:  

            with codecs.open(os.path.join(pname, dname), 'r', 'utf-8') as f:
                question = f.readline()
                question = question.split(' ')
                question_array = [word_vocab.feed(c) for c in ['{'] + question + ['}']]
                if len(question_array) > max_sent_length:
                    question_array = question_array[:max_sent_length]
                lines = f.read().split('\n')
                word_doc = []
                label_doc = []
                sent_doc = []

                for line in lines:
                    line = line.strip()
                    line = line.replace('}', '').replace('{', '').replace('|', '')
                    line = line.replace('<unk>', ' | ')
                    if line == '':
                        continue
                    sent, label = line.split('\t')  
                    label_doc.append(label)
                    sent_doc.append(sent)
                    sent = sent.split(' ')

                    if len(sent) > max_sent_length - 2:  # space for 'start' and 'end' words
                        sent = sent[:max_sent_length-2]

                    word_array = [word_vocab.feed(c) for c in ['{'] + sent + ['}']]
                    
                    word_doc.append(word_array)
  
                if len(word_doc) > max_doc_length:
                    word_doc = word_doc[:max_doc_length]
                    label_doc = label_doc[:max_doc_length]
                    sent_doc = sent_doc[:max_doc_length]

                actual_max_doc_length = max(actual_max_doc_length, len(word_doc))

                word_tokens[fname].append(word_doc)
                labels[fname].append(label_doc)
                questions[fname].append(question_array)
                sentences[fname].append(sent_doc)

    word_vocab.save(filename='vocab')
    assert actual_max_doc_length <= max_doc_length

    print()
    print('actual longest document length is:', actual_max_doc_length)
    print('size of word vocabulary:', word_vocab.size)
    print('number of tokens in train:', len(word_tokens['train']))
    print('number of tokens in valid:', len(word_tokens['valid']))
    print('number of tokens in test:', len(word_tokens['test']))

    # now we know the sizes, create tensors
    word_tensors = {}
    label_tensors = {}
    question_tensors = {}
    for fname in data_set:
        word_tensors[fname] = np.zeros([len(word_tokens[fname]), actual_max_doc_length, max_sent_length], dtype=np.int32)
        label_tensors[fname] = np.zeros([len(labels[fname]), actual_max_doc_length], dtype=np.int32)
        question_tensors[fname] = np.zeros([len(questions[fname]), max_sent_length], dtype=np.int32)
 
        for i, word_doc in enumerate(word_tokens[fname]):
            for j, word_array in enumerate(word_doc):
                word_tensors[fname][i][j][0:len(word_array)] = word_array

        for i, label_doc in enumerate(labels[fname]):
            label_tensors[fname][i][0:len(label_doc)] = label_doc

        for i, question in enumerate(questions[fname]):
            question_tensors[fname][i][0:len(question)] = question

    return word_vocab, word_tensors, actual_max_doc_length, label_tensors, question_tensors, sentences


class DataReader:

    def __init__(self, word_tensor, label_tensor, question_tensor, batch_size):

        length = word_tensor.shape[0]

        doc_length = word_tensor.shape[1]
        sent_length = word_tensor.shape[2]

        # round down length to whole number of slices

        clipped_length = int(length / batch_size) * batch_size
        word_tensor = word_tensor[:clipped_length]
        label_tensor = label_tensor[:clipped_length]
        question_tensor = question_tensor[:clipped_length]

        x_batches = word_tensor.reshape([batch_size, -1, doc_length, sent_length])
        y_batches = label_tensor.reshape([batch_size, -1, doc_length])
        q_batches = question_tensor.reshape([batch_size, -1, sent_length])

        x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        q_batches = np.transpose(q_batches, axes=(1, 0, 2))

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        self._q_batches = list(q_batches)
        assert len(self._x_batches) == len(self._y_batches) == len(self._q_batches)
        self.length = len(self._y_batches)
        self.batch_size = batch_size
        self.max_sent_length = sent_length

    def iter(self):

        for x, y, q in zip(self._x_batches, self._y_batches, self._q_batches):
            yield x, y, q



if __name__ == '__main__':

    vocab, word_tensors, max_length, label_tensors, question_tensors = load_data('data/demo', 5, 10)

    count = 0
    for x, y, q in DataReader(word_tensors['valid'], label_tensors['valid'], question_tensors['valid'], 6).iter():
        count += 1
        print (x.shape, y.shape, q.shape)
        if count > 0:
            break



