# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D #, Convolution1D, MaxPooling1D
import numpy as np
from six.moves import range


# Parameters for the model and dataset
TRAINING_SIZE = 50000
CART_SIZE = 3
#MAXLEN_q = (4+1+5+1)*CART_SIZE # ( category_id+price, )*CART_SIZE
MAXLEN_q = (4+1)*CART_SIZE # ( category_id, )*CART_SIZE
#结果包含多个分类
#MAXLEN_a = (4+1)*CART_SIZE # ( category_id, )*CART_SIZE
#结果包含1个分类
MAXLEN_a = 4 # category_id

batch_size = 128
nb_classes = 10
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def load_data(file_name):
    import json

    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    all_data = []
    for i in lines:
        x = json.loads(i)

        # 忽略sku数大于CART_SIZE的数据
        if len(x['last'][1])>CART_SIZE:
            continue

        # 去掉价格
        q2 = [qq[0] for qq in x['last'][1]]
        q2 = list(set(q2))
        q2 = sorted(q2)

        #结果包含多个分类
        #all_data.append((x['last'][1], x['next'][1]))
        #结果包含1个分类
        for j in x['next'][1]:
            #all_data.append((x['last'][1], [j]))
            all_data.append((q2, [j]))
    
    return all_data




chars = '0123456789+, '
ctable = CharacterTable(chars, MAXLEN_q)

def main():

    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    train_data = load_data('2017-01-12.dat')
    train_data = train_data[:TRAINING_SIZE] # for test
    for x in train_data:
        #query = ','.join(['%04d+%05d'%(i[0],i[1]) for i in x[0]][:CART_SIZE])
        query = ','.join(['%04d'%i for i in x[0]][:CART_SIZE]) # 不含价格信息
        ans = ','.join(['%04d'%i for i in x[1]][:CART_SIZE])
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    #print(questions)
    #print(expected)

    print('Vectorization...')
    X = np.zeros((len(questions), MAXLEN_q, len(chars), 1))
    y = np.zeros((len(questions), MAXLEN_a, len(chars)))
    for i, sentence in enumerate(questions):
        X[i] = ctable.encode(sentence, maxlen=MAXLEN_q)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, maxlen=MAXLEN_a)

    # Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Explicitly set apart 10% for validation data that we never train over
    split_at = len(X) - len(X) / 10
    (X_train, X_val) = (slice_X(X, 0, split_at), slice_X(X, split_at))
    (y_train, y_val) = (y[:split_at], y[split_at:])

    print(X_train.shape)
    print(y_train.shape)

    #return X_train, y_train


    print('Build model...')
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(MAXLEN_q, len(chars), 1) ))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_val, y_val))
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    main()
