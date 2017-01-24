# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.models import load_model
from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Reshape
import numpy as np
from six.moves import range


# Parameters for the model and dataset
TRAINING_SIZE = 100000
# Try replacing GRU, or SimpleRNN or LSTM
RNN = recurrent.SimpleRNN
HIDDEN_SIZE = 64
BATCH_SIZE = 64
LAYERS = 1
CART_SIZE = 3
#MAXLEN_q = (4+1+5+1)*CART_SIZE # ( category_id+price, )*CART_SIZE
MAXLEN_q = (4+1)*CART_SIZE # ( category_id, )*CART_SIZE
#结果包含多个分类
#MAXLEN_a = (4+1)*CART_SIZE # ( category_id, )*CART_SIZE
#结果包含1个分类
MAXLEN_a = 4 # category_id


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

    print('Vectorization...')
    X = np.zeros((len(questions), MAXLEN_q, len(chars)))
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

    #print(questions)
    #print(expected)
    #print(X_train)
    #print(y_train)
    #return X_train, y_train

    print('Build model...')
    model = Sequential()
    #model.add(Reshape((CART_SIZE, MAXLEN, len(chars)), input_shape=(CART_SIZE*MAXLEN*len(chars),)))
    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
    # note: in a situation where your input sequences have a variable length,
    # use input_shape=(None, nb_feature).
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN_q, len(chars))))
    # For the decoder's input, we repeat the encoded input for each time step
    model.add(RepeatVector(MAXLEN_a))
    # The decoder RNN could be multiple layers stacked or a single layer
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    # For each of step of the output sequence, decide which character should be chosen
    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, 50):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=1,
                  validation_data=(X_val, y_val))
        ###
        # Select samples from the validation set at random so we can visualize errors
        for i in range(5):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            q = ctable.decode(rowX[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q)
            print('T', correct)
            print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close, guess)
            print('---')

    model.save('my_model_rnn.h5')

if __name__ == "__main__":
    #main()

    print('Loading model ... ')
    model = load_model('my_model_rnn.h5')

    print('Guessing ... ')
    test_q = ['0690','0694']

    test_str = ','.join(test_q[:CART_SIZE])
    test_val = ctable.encode(test_str, maxlen=MAXLEN_q)

    preds = model.predict_classes(np.array([test_val]), verbose=0)
    guess = ctable.decode(preds[0], calc_argmax=False)

    print('Question:', test_str)
    print('Ans:', guess)
