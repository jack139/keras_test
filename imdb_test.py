#!/usr/bin/env python
# -*- coding: utf-8 -*-


from keras.datasets import imdb
import numpy as np

from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers

# 加载电影分类数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#将整数序列编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#将标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 三层网络 模型定义
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#编译模型
model.compile(#optimizer='rmsprop',
			  optimizer=optimizers.RMSprop(lr=0.0005),
			  loss='binary_crossentropy',
			  #loss='mse',
			  #metrics=['accuracy']
			  metrics=['acc']
			  #metrics=[metrics.binary_accuracy]
			  )

#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


#训练模型
model.fit(partial_x_train,
		  partial_y_train,
		  epochs=10,
		  batch_size=512,
		  validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)

print(results)

# 使用训练好的模型进行预测
#model.predict(x_test)


#if __name__ == "__main__":
