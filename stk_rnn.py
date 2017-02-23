# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.models import load_model
from keras.engine.training import slice_X
from keras.layers import TimeDistributed, RepeatVector, recurrent, Reshape
from keras.optimizers import SGD
import numpy as np
from six.moves import range


# Parameters for the model and dataset
TRAINING_SIZE = 10000
# Try replacing GRU, or SimpleRNN or LSTM
RNN = recurrent.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

INPUT_LENGTH = 10 # 输入数据长度 7天
MAXLEN_q = INPUT_LENGTH # 7天数据
MAXLEN_a = 1 # 1天数据
MAXW_q = 8 # 星期几，涨跌%
MAXW_a = 5 #


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

#日期,股票代码,名称,收盘价,最高价,最低价,开盘价,前收盘,涨跌额,涨跌幅,换手率,成交量,成交金额
#2017-02-16,'000333,美的集团,32.03,32.1,30.93,31,31.06,0.97,3.123,0.649,41327408,1302878001.71
#2017-02-15,'000333,美的集团,31.06,31.45,30.7,30.83,30.69,0.37,1.2056,0.4778,30427042,947849153.89
#2017-02-14,'000333,美的集团,30.69,31.29,30.6,31,30.91,-0.22,-0.7117,0.4165,26523718,818126119.56
#2017-02-13,'000333,美的集团,30.91,30.94,30.05,30.05,30.12,0.79,2.6228,0.7011,44650847,1368505944
#2017-02-10,'000333,美的集团,30.12,30.48,29.99,30.1,30.02,0.1,0.3331,0.4238,26991885,814459157.03
#2017-02-09,'000333,美的集团,30.02,30.12,29.87,29.94,29.88,0.14,0.4685,0.3942,25101671,753527779.16
#2017-02-08,'000333,美的集团,29.88,30.07,29.58,29.65,29.65,0.23,0.7757,0.4023,25621590,763569863.39
#2017-02-07,'000333,美的集团,29.65,30.17,29.54,30,29.99,-0.34,-1.1337,0.5088,32400398,965184287.41

# question 星期几，涨跌幅
#(1,3.123)
#(2,1.2056)
#(3,-0.7117)
#(4,2.6228)
#(5,0.3331)
#(6,0.4685)
#(7,0.7757)

# answer 涨跌幅
#(-1.1337)


def gen_data(line):
    import time
    x = line.split(',')

    date = x[0] # 日期
    wday = time.strptime(date,'%Y-%m-%d').tm_wday # 0-6, 0-Monday
    if x[9]=='None':
        zhang_die = 0.0
        percent = 0.0
    else:
        zhang_die = float(x[8])
        #percent = float(x[9])/100.0 # 涨跌幅
        percent = float(x[9]) # 涨跌幅


    #return [float(wday)/10.0, percent] # 都用浮点数表示
    return [
        float(wday), # 星期几
        percent, # 涨跌幅
        float(x[3]), # 收盘价 
        float(x[4]), # 最高价 
        float(x[5]), # 最低价 
        float(x[6]), # 开盘价 
        zhang_die, # 涨跌额
        float(x[10]), # 成交量／换手率
    ] 


def load_data(file_name):
    import json

    f = open(file_name, 'r')
    lines = f.readlines()
    f.close()

    all_data = []
    for index, i in enumerate(lines):
        if index<INPUT_LENGTH+1: # 从第8行开始， 0行是标题
            continue

        question = []
        for j in xrange(index-INPUT_LENGTH, index):
            question.append(gen_data(lines[j]))

        _answer = gen_data(lines[index])

        if _answer[1]<=-5.0:
            answer = 0
        elif _answer[1]<0.0 and _answer[1]>-5.0:
            answer = 1
        elif _answer[1]==0.0:
            answer = 2
        elif _answer[1]>0.0 and _answer[1]<5.0:
            answer = 3
        else:
            answer = 4

        all_data.append([question, answer])
    
    return all_data



def main():

    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    train_data = load_data('000333.csv')
    train_data = train_data[:TRAINING_SIZE] # for test
    for x in train_data:
        #print(x)
        X = np.zeros((MAXLEN_q, MAXW_q))
        for i, c in enumerate(x[0]):
            X[i] = c
        questions.append(X)

        Y = np.zeros((MAXLEN_a, MAXW_a))
        Y[0, x[1]] = 1
        expected.append(Y)
    print('Total addition questions:', len(questions))
    #print(questions)
    #print(expected)

    print('Vectorization...')
    X = np.zeros((len(questions), MAXLEN_q, MAXW_q))
    y = np.zeros((len(questions), MAXW_a))
    for i, sentence in enumerate(questions):
        X[i] = sentence
    for i, sentence in enumerate(expected):
        y[i] = sentence

    # Shuffle (X, y) 
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

    #print(X_train)
    #print(y_train)
    #return X_train, y_train

    print('Build model...')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN_q, MAXW_q)))
    model.add(Dropout(0.5))
    model.add(Dense(HIDDEN_SIZE, activation='tanh'))
    model.add(Dense(MAXW_a, activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    # Train the model each generation and show predictions against the validation dataset
    for iteration in range(1, 10):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=10,
                  validation_data=(X_val, y_val))

        ###
        # Select samples from the validation set at random so we can visualize errors
        for i in range(5):
            ind = np.random.randint(0, len(X_val))
            rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
            preds = model.predict_classes(rowX, verbose=0)
            #print(rowX)
            #print(rowy)
            #print(preds)
            q = rowX[0]
            correct = rowy[0]
            guess = preds[0]
            print('Q', q)
            print('T', correct)
            print('G', preds)
            print(colors.ok + '☑' + colors.close if correct[guess] > 0 else colors.fail + '☒' + colors.close, guess)
            print('---')

    model.save('stk_rnn.h5')

if __name__ == "__main__":
    #main()

    print('Loading model ... ')
    model = load_model('stk_rnn.h5')

    print('Guessing ... ')
    test_q = ['0690','0694']

    test_str = ','.join(test_q[:CART_SIZE])
    test_val = ctable.encode(test_str, maxlen=MAXLEN_q)

    preds = model.predict_classes(np.array([test_val]), verbose=0)
    guess = ctable.decode(preds[0], calc_argmax=False)

    print('Question:', test_str)
    print('Ans:', guess)
