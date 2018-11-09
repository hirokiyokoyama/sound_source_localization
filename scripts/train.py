#!/usr/bin/env python

import matplotlib.pyplot as plt
from audio_common_msgs.msg import AudioData
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from nets import conv_deconv
from dataset import get_recorded_dataset
import sys

CHANNELS = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE/5

class Trainer:
    def __init__(self, channels):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.int16, shape=[None, None, channels])
            inputs = tf.cast(self._inputs, tf.float32) / 32768
            inputs = tf.transpose(inputs, [0,2,1])
            # [N,C,T]
            spectrogram = tf.contrib.signal.stft(inputs, frame_length=480, frame_step=160)
            # [N,C,T,F]
            self._spectrogram = tf.transpose(spectrogram, [0,2,3,1])
            # [N,T,F,C]
            self._features = tf.concat([tf.real(self._spectrogram), tf.imag(self._spectrogram)], -1)
            self._hidden, self._logits = conv_deconv(self._features)
            # [N,T,W,W]
            self._sound_source_map = tf.nn.sigmoid(self._logits)
            
            weights = tf.abs(self._spectrogram)
            weights = tf.reduce_mean(weights, -1, keepdims=True)
            weights = tf.reduce_mean(weights, -1, keepdims=True) # [N,T,1,1]
            self._sess = tf.Session(graph=self._graph)

            self._init_op = tf.global_variables_initializer()

    def initialize(self):
        self._sess.run(self._init_op)

    def spectrogram(self, inputs):
        return self._sess.run(self._spectrogram, {self._inputs: inputs})

    def sound_source_map(self, inputs):
        return self._sess.run(self._sound_source_map, {self._inputs: inputs})

trainer = Trainer(CHANNELS)
trainer.initialize()

dataset = get_recorded_dataset(sys.argv[1:])
for data in dataset:
    rate, d = wavfile.read(data['recorded_sound_file'])
    assert rate == SAMPLE_RATE
    assert d.shape[1] == CHANNELS
    m = np.abs(trainer.spectrogram([d])[0])

    rate, _d = wavfile.read(data['sound_file'])
    _d = np.tile(np.expand_dims(_d, -1), [1,CHANNELS])
    _m = np.abs(trainer.spectrogram([_d])[0])

    import matplotlib.pyplot as plt
    plt.clf()
    
    plt.plot(m.mean(2).mean(1))
    plt.plot(_m.mean(2).mean(1))
    plt.ylim([0,1])
    plt.pause(1)
