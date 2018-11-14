import rospy
import threading
from audio_common_msgs.msg import AudioData
from utils import FrequencyMeter
import numpy as np
import pyaudio

class SoundListener:
    def __init__(self, buf):
        self._freq_meter = FrequencyMeter(100)
        self._sub = rospy.Subscriber('audio', AudioData, self._cb, queue_size=100)
        self._buffer = buf
        
    def _cb(self, msg):
        data = np.fromstring(msg.data, dtype=np.int16)
        self._buffer.append(data)
        
        f = self._freq_meter.tap()
        if f is not None:
            print '{} Hz'.format(f)

class ListBuffer:
    def __init__(self, size, channels):
        self._buf = np.zeros([size, channels], dtype=np.int16)
        self._pointer = -1
        self._lock = threading.Lock()

    def append(self, data):
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

class QueueBuffer:
    def __init__(self, capacity, channels):
        self._buf = np.zeros([capacity, channels], dtype=np.int16)
        self._pointer = 0
        self._size = 0
        self._lock = threading.Lock()

    def append(self, data):
        data = data.reshape(-1, self._buf.shape[1])
        cap = self._buf.shape[0]
        if data.shape[0] > cap:
            data = data[-cap:]
        with self._lock:
            begin = self._pointer
            end = begin+data.shape[0]
            if end < cap:
                self._buf[begin:end] = data
                self._pointer = end
            else:
                a = cap - begin
                b = end - cap
                self._buf[begin:] = data[:a]
                self._buf[:b] = data[a:]
                self._pointer = b
            self._size = min(self._size + data.shape[0], cap)
            
    def get_latest(self, frames):
        with self._lock:
            frames = min(self._size, frames)
            if frames <= self._pointer:
                data = self._buf[self._pointer-frames:self._pointer].copy()
            else:
                a = frames - self._pointer
                data1 = self._buf[-a:].copy()
                data2 = self._buf[:self._pointer].copy()
                data = np.concatenate([data1, data2], 0)
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
        assert data.dtype == np.int16 or data.dtype == np.float32 or data.dtype == np.float64
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = np.int16(data*32768.)
        duration = data.shape[0]/float(self._sample_rate)
        self._stream.start_stream()
        self._stream.write(data.tostring())
        rospy.sleep(duration)
        self._stream.stop_stream()

    def __del__(self):
        self._stream.close()
        self._pa.terminate()
