#!/usr/bin/env python

import matplotlib.pyplot as plt
from audio_common_msgs.msg import AudioData
import rospy
import numpy as np
import tensorflow as tf
from utils import FrequencyMeter
from sound_source_localization.srv import Synchronize, SynchronizeResponse
import threading
from scipy.io import wavfile

CHANNELS = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE*10

class SoundProcessor:
    def __init__(self, channels):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._inputs = tf.placeholder(tf.int16, shape=[None, channels])
            inputs = tf.cast(self._inputs, tf.float32) / 32768
            inputs = tf.transpose(inputs, [1,0])
            spectrogram = tf.contrib.signal.stft(inputs, frame_length=480, frame_step=160)
            self._spectrogram = tf.transpose(spectrogram, [1,2,0])
            self._features = tf.concat([tf.real(self._spectrogram), tf.imag(self._spectrogram)], -1)
            print self._features
            self._sess = tf.Session(graph=self._graph)

    def process(self, inputs):
        return self._sess.run(self._spectrogram, {self._inputs: inputs})

class SoundListener:
    def __init__(self, buffer_size, channels):
        self._freq_meter = FrequencyMeter(100)
        self._sub = rospy.Subscriber('audio', AudioData, self._cb)

        self._data = np.zeros([buffer_size, channels], dtype=np.int16)
        self._flag = np.zeros([buffer_size], dtype=np.bool)
        self._pointer = 0
        
    def _cb(self, msg):
        data = np.fromstring(msg.data, dtype=np.int16)
        data = data.reshape(-1, self._data.shape[1])
        begin = self._pointer
        end = min(self._pointer+data.shape[0], self._data.shape[0])
        self._data[begin:end,:] = data[:end-begin,:]
        self._flag[begin:end] = np.abs(data).max() > 16384
        self._pointer = end
        if self._pointer == self._data.shape[0]:
            self._pointer = 0
        
        f = self._freq_meter.tap()
        if f is not None:
            print '{} Hz'.format(f)

    def check(self):
        if np.any(self._flag):
            return self._data.copy()
        return None

freq_meter = FrequencyMeter(100)
sl = SoundListener(BUFFER_SIZE, CHANNELS)
sp = SoundProcessor(CHANNELS)

def show():
    f = freq_meter.tap()
    if f is not None:
        print 'show: {} Hz'.format(f)
    d = sl.check()
    if d is None:
        return
    spectrogram = sp.process(d)
    plt.clf()
    spec = np.abs(spectrogram[:,:,:3].transpose(1,0,2))
    spec /= spec.max()
    plt.imshow(spec, origin='lower')
    plt.pause(.001)

class Synchronizer:
    def __init__(self):
        self._iteration_local = 0
        self._iteration_remote = 0
        rospy.Service('/synchronize', Synchronize, self._srv_cb)
        self._srv = rospy.ServiceProxy('/bridged/synchronize', Synchronize)
        rospy.loginfo('Waiting for another robot.')
        self._srv.wait_for_service()
        self._cond = threading.Condition()
        
    def _srv_cb(self, req):
        with self._cond:
            self._iteration_remote = req.iteration
            self._cond.notify()
        return SynchronizeResponse(self._iteration_local)

    def next(self):
        self._iteration_local += 1
        self._srv(self._iteration_local)
        with self._cond:
            assert self._iteration_remote <= self._iteration_local, 'Something happened!'
            while self._iteration_remote < self._iteration_local:
                self._cond.wait()

    @property
    def iteration(self):
        return self._iteration_local

rospy.init_node('ssl_record')
sync = Synchronizer()

rate = rospy.Rate(1)
while not rospy.is_shutdown():
    sync.next()
    print 'Iteration %d' % sync.iteration
    #show()
    rate.sleep()
    #wavfile.write(filename, SAMPLE_RATE, y3)
    
