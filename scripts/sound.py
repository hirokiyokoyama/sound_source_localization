import rospy
import threading
from audio_common_msgs.msg import AudioData
from utils import FrequencyMeter
import numpy as np
import pyaudio

class SoundListener:
    def __init__(self, buffer_size, channels):
        self._freq_meter = FrequencyMeter(100)
        self._sub = rospy.Subscriber('audio', AudioData, self._cb, queue_size=100)

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
        if self._pointer < 0:
            raise ValueError('Recording not started yet.')
        with self._lock:
            data = self._buf[:self._pointer].copy()
            self._pointer = -1
        return data

class SoundPlayer:
    def __init__(self, channels=1, sample_rate=16000):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=channels,
                                     rate=sample_rate,
                                     output=True)
        self._stream.stop_stream()
        self._channels = channels
        self._sample_rate = sample_rate
        
    def play(self, data):
        if len(data.shape) == 1:
            data = np.expand_dims(data, 1)
        assert len(data.shape) == 2 and data.shape[1] == self._channels
        assert data.dtype == np.int16
        duration = data.shape[0]/float(self._sample_rate)
        self._stream.start_stream()
        self._stream.write(data.tostring())
        rospy.sleep(duration)
        self._stream.stop_stream()

    def __del__(self):
        self._stream.close()
        self._pa.terminate()
