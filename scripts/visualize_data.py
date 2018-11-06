#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
from scipy.io import wavfile
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from sound import SoundPlayer
import tf

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
            self._sess = tf.Session(graph=self._graph)

    def process(self, inputs):
        return self._sess.run(self._spectrogram, {self._inputs: inputs})

data_dir = os.path.abspath(sys.argv[1])
data_num = sys.argv[2]
rate, sound = wavfile.read(os.path.join(data_dir, 'sound_'+data_num+'.wav'))
channels = sound.shape[1]

mapmsg = OccupancyGrid()
with open(os.path.join(data_dir, 'map.msg'), 'rb') as f:
    mapmsg.deserialize(f.read())
#lscanmsg = LaserScan()
#with open(os.path.join(data_dir, 'local_scan_'+data_num+'.msg'), 'rb') as f:
#    lscanmsg.deserialize(f.read())
#rscanmsg = LaserScan()
#with open(os.path.join(data_dir, 'remote_scan_'+data_num+'.msg'), 'rb') as f:
#    rscanmsg.deserialize(f.read())
lposemsg = PoseWithCovarianceStamped()
with open(os.path.join(data_dir, 'local_pose_'+data_num+'.msg'), 'rb') as f:
    lposemsg.deserialize(f.read())
rposemsg = PoseWithCovarianceStamped()
with open(os.path.join(data_dir, 'remote_pose_'+data_num+'.msg'), 'rb') as f:
    rposemsg.deserialize(f.read())

sp = SoundProcessor(channels)
spectrogram = sp.process(sound)
plt.figure()
for i in range(channels):
    plt.subplot(1,channels,i+1)
    spec = np.abs(spectrogram[:,:,i].T)
    spec /= spec.max()
    plt.imshow(spec, origin='lower')

#player = SoundPlayer(channels=channels, sample_rate=rate)
#player.play(sound)

_mapimg = np.array(mapmsg.data).reshape(mapmsg.info.height, mapmsg.info.width)
mapimg = np.ones([mapmsg.info.height,mapmsg.info.width,3], dtype=np.uint8) * 255
mapimg[np.where(_mapimg>0)] = 0
print mapmsg.info.resolution
print mapmsg.info.origin

q = mapmsg.info.origin.orientation
rot = tf.transformations.quaternion_matrix([q.x,q.y,q.z,q.w])
p = mapmsg.info.origin.position
trans = tf.transformations.translation_matrix([p.x,p.y,p.z])
scale = tf.transformations.scale_matrix(mapmsg.info.resolution)
pixel2map = np.matmul(trans, np.matmul(rot, scale))
map2pixel = np.linalg.inv(pixel2map)

plt.figure()
plt.imshow(mapimg, origin='lower')
plt.show()
