
import pickle
import numpy as np
from keras.models import Sequential
import gensim
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from keras import optimizers
import math
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
def step_decay(epoch):
	initial_lrate = 2
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


with open('conversation6.pickle','rb') as f:
    vec_x, vec_y = pickle.load(f)

vec_x = np.array(vec_x, dtype=np.float64)
vec_y = np.array(vec_y, dtype=np.float64)

x_train, x_test, y_train, y_test = train_test_split(vec_x, vec_y, test_size=0.2,random_state=1)

model = Sequential()
model.add(LSTM(units=300, input_shape=x_train.shape[1:], return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
               inner_init='glorot_normal', activation='sigmoid'))
model.add(LSTM(units=300,  return_sequences=True, init='glorot_normal',
           inner_init='glorot_normal', activation='sigmoid'))


decay_rate =0.6/800
momentum = 0.8
#optimizers.SGD(lr=2, momentum=0.8, decay=decay_rate, nesterov=False)
model.compile(loss='cosine_proximity', optimizer='SGD', metrics=['accuracy'])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), callbacks=callbacks_list,)
model.save('LSTM1a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test),callbacks=callbacks_list)
model.save('LSTM2a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM3a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM4a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM5a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM6a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM7a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM8a.h5');
model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTM9a.h5');
model.fit(x_train, y_train, epochs=2500, validation_data=(x_test, y_test))
model.save('LSTM10a.h5');
model.fit(x_train, y_train, epochs=3500, validation_data=(x_test, y_test))
model.save('LSTM11a.h5');
predictions = model.predict(x_test)
mod = gensim.models.Word2Vec.load('word2vec.bin');
[mod.most_similar([predictions[10][i]])[0] for i in range(15)]
