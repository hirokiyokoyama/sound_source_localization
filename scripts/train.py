#!/usr/bin/env python

import matplotlib.pyplot as plt
from audio_common_msgs.msg import AudioData
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from nets import conv_deconv
from dataset import get_recorded_dataset
from utils import axis_vector
import os, sys

CHANNELS = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE/5
NUM_DECONV = 2
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'models')
LEARNING_RATE = 0.001
print MODEL_DIR

class Trainer:
    def __init__(self, channels, frame_length=480, frame_step=160):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.float32, shape=[None,None,channels])
            self._labels = tf.placeholder(tf.float32, shape=[None,None,None,None])
            
            inputs = tf.transpose(self._inputs, [0,2,1])
            # [N,C,T]
            spectrogram = tf.contrib.signal.stft(inputs,
                                                 frame_length,
                                                 frame_step,
                                                 pad_end=True)
            # [N,C,T,F]
            self._spectrogram = tf.transpose(spectrogram, [0,2,3,1])
            # [N,T,F,C]
            self._features = tf.concat([tf.real(self._spectrogram), tf.imag(self._spectrogram)], -1)
            self._logits = conv_deconv(self._features, num_deconv=NUM_DECONV)
            # [N,T,W,W]
            self._sound_source_map = tf.nn.sigmoid(self._logits)

            labels = self._labels
            #shape = tf.shape(self._labels)
            #size = 3*2**NUM_DECONV
            #labels = tf.reshape(self._labels, [shape[0]*shape[1], shape[2], shape[3], 1])
            #labels = tf.image.resize_images(labels, [size,size])
            #labels = tf.reshape(labels, [shape[0],shape[1],size,size])
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=labels)
            
            weights = tf.abs(self._spectrogram)
            weights = tf.reduce_mean(weights, 2, keepdims=True)
            weights = tf.reduce_mean(weights, 3, keepdims=True)
            # [N,T,1,1]
            self._losses = losses * weights
            opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            self._train_op = opt.minimize(tf.reduce_mean(self._losses))
            
            self._init_op = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._sess = tf.Session(graph=self._graph)

    def initialize_variables(self):
        self._sess.run(self._init_op)

    def restore_variables(self, ckpt):
        self._saver.restore(self._sess, ckpt)

    def spectrogram(self, inputs):
        return self._sess.run(self._spectrogram, {self._inputs: inputs})

    def sound_source_map(self, inputs):
        return self._sess.run(self._sound_source_map, {self._inputs: inputs})

    def train(self, inputs, labels):
        _, losses = self._sess.run([self._train_op, self._losses],
                                   {self._inputs: inputs, self._labels: labels})
        return losses

trainer = Trainer(CHANNELS)
latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
if latest_ckpt is not None:
    print 'Restoring from %s.' % latest_ckpt
    trainer.restore_variables(latest_ckpt)
else:
    print 'Starting with a new model.'
    trainer.initialize_variables()

dataset = get_recorded_dataset(sys.argv[1:])
for data in dataset:
    rate, d = wavfile.read(data['recorded_sound_file'])
    assert rate == SAMPLE_RATE
    assert d.shape[1] == CHANNELS
    m = np.abs(trainer.spectrogram([d / 32768.])[0])

    print data['text']
    mic_pos = data['mic_pose'].pose.position
    mic_pos = np.array([mic_pos.x, mic_pos.y, mic_pos.z])
    mic_vec = axis_vector(data['mic_pose'].pose.orientation, 2)
    self_vec = axis_vector(data['self_pose'].pose.orientation, 0)
    speaker_pos = data['speaker_pose'].pose.position
    speaker_pos = np.array([speaker_pos.x, speaker_pos.y, speaker_pos.z])
    speaker_vec = axis_vector(data['speaker_pose'].pose.orientation, 2)
    mic_ori = np.arctan2(mic_vec[1], mic_vec[0])
    d = speaker_pos-mic_pos
    dist = np.hypot(d[0], d[1])
    theta = np.arctan2(d[1], d[0]) - mic_ori
    print mic_vec, self_vec, dist, theta/np.pi*180

    rate, _d = wavfile.read(data['sound_file'])
    
    _d = np.tile(np.expand_dims(_d / 32768., -1), [1,CHANNELS])
    _m = np.abs(trainer.spectrogram([_d])[0])

    labels = np.zeros([1,d.shape[0]/160,3*2**NUM_DECONV,3*2**NUM_DECONV])
    #print trainer.train([d/32768.], labels)
    
    import matplotlib.pyplot as plt
    plt.clf()
    
    plt.plot(m.mean(2).mean(1))
    plt.plot(_m.mean(2).mean(1))
    plt.ylim([0,1])
    plt.pause(1)
