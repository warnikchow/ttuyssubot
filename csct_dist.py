import numpy as np
import sys

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

import fasttext

print('\nImporting dictionaries...')

model_drama = fasttext.load_model('vectors/model_drama.bin')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, LSTM, GRU, SimpleRNN, Dense, Lambda
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import keras.layers as layers

from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)

from keras.preprocessing import sequence
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization


from keras.models import load_model
model_corr100 = load_model('modelcwsre/rnnconvdnn100_sigmoid_concat-28-0.9885.hdf5')

def pred_correction_rnn(sent,model,dic,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    rnn  = np.zeros((1,maxlen,wdim))
    charcount = -1
    for j in range(len(sent)):
      if j<maxlen and sent[j]!=' ':
        charcount=charcount+1
        conv[0][charcount,:,0]=dic[sent[j]]
        rnn[0][charcount,:]=dic[sent[j]]
    z = model.predict([conv,rnn])[0]
    sent_raw = ''
    count_char=-1
    lastpoint=-1
    lastchar=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>threshold:
          sent_raw = sent_raw+' '
          if j<len(sent)-2:
            lastpoint=len(sent_raw)
            lastchar=j
    return sent_raw, lastpoint, lastchar

def correct(s):
    if len(s)<100:
      temp,lp,lc = pred_correction_rnn(s,model_corr100,model_drama,100,100)
      z = temp
    else:
      z=''
      start=0
      while start<len(s):
        if start+50<len(s):
          temp,lp,lc =pred_correction_rnn(s[start:start+50],model_corr100,model_drama,100,100)
          temp=temp[:lp]
        else:
          temp,lp,lc =pred_correction_rnn(s[start:],model_corr100,model_drama,100,100)
          lc = 50
        z = z+temp
        start=start+lc+1
    return z
