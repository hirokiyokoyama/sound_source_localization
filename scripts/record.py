#!/usr/bin/env python

from audio_common_msgs.msg import AudioData
import rospy
import numpy as np
from utils import FrequencyMeter
from sound_source_localization.srv import Synchronize, SynchronizeResponse
import threading
from scipy.io import wavfile
import os

class SoundListener:
    def __init__(self, buffer_size, channels):
        self._freq_meter = FrequencyMeter(100)
        self._sub = rospy.Subscriber('audio', AudioData, self._cb)

        self._buf = np.zeros([buffer_size, channels], dtype=np.int16)
        self._pointer = -1
        self._lock = threading.Lock()
        
    def _cb(self, msg):
        data = np.fromstring(msg.data, dtype=np.int16)
        data = data.reshape(-1, self._buf.shape[1])
        begin = 0
        end = 0
        with self._lock:
            if self._pointer >= 0:
                begin = self._pointer
                end = min(begin+data.shape[0], self._buf.shape[0])
                self._buf[begin:end,:] = data[:end-begin,:]
                self._pointer = end
        if end == self._buf.shape[0]:
            rospy.logwarn('Buffer overrun.')
        
        f = self._freq_meter.tap()
        if f is not None:
            print '{} Hz'.format(f)

    def start(self):
        with self._lock:
            self._pointer = 0
            
    def stop(self):
        with self._lock:
            self._pointer = -1
            return self._buf.copy()

class Synchronizer:
    def __init__(self):
        self._iteration_local = 0
        self._iteration_remote = 0
        self._cond = threading.Condition()
        rospy.Service('/synchronize', Synchronize, self._srv_cb)
        self._srv = rospy.ServiceProxy('/bridged/synchronize', Synchronize)
        rospy.loginfo('Waiting for another robot.')
        self._srv.wait_for_service()
        
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
save_dir = rospy.get_param('~save_dir')
channels = rospy.get_param('ssl/channels')
sample_rate = rospy.get_param('ssl/sample_rate')
sl = SoundListener(sample_rate*10, channels)
sync = Synchronizer()

rate = rospy.Rate(1)
while not rospy.is_shutdown():
    sync.next()
    print 'Iteration %d' % sync.iteration
    sl.start()
    rate.sleep()
    data = sl.stop()
    filename = os.path.join(save_dir, 'sound_{:04d}.wav'.format(sync.iteration))
    wavfile.write(filename, sample_rate, data)
