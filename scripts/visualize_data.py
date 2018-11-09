#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
from scipy.io import wavfile
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from sound import SoundPlayer
from train import Trainer
from tf import transformations
import yaml

def scan2points(scanmsg, posemsg):
    N = len(scanmsg.ranges)
    points = np.zeros([4,N])
    angles = scanmsg.angle_increment*np.arange(N) + scanmsg.angle_min
    points[0,:] = np.cos(angles) * np.array(scanmsg.ranges)
    points[1,:] = np.sin(angles) * np.array(scanmsg.ranges)
    points[3,:] = 1.
    o = posemsg.orientation
    o = transformations.quaternion_matrix([o.x, o.y, o.z, o.w])
    points = np.matmul(o, points)
    points[0,:] += posemsg.position.x
    points[1,:] += posemsg.position.y
    points[2,:] += posemsg.position.z
    return points

data_dir = os.path.abspath(sys.argv[1])
data_num = sys.argv[2]
rate, sound = wavfile.read(os.path.join(data_dir, 'sound_'+data_num+'.wav'))
channels = sound.shape[1]

mapmsg = OccupancyGrid()
with open(os.path.join(data_dir, 'map.msg'), 'rb') as f:
    mapmsg.deserialize(f.read())
lscanmsg = LaserScan()
with open(os.path.join(data_dir, 'self_scan_'+data_num+'.msg'), 'rb') as f:
    lscanmsg.deserialize(f.read())
rscanmsg = LaserScan()
with open(os.path.join(data_dir, 'other_scan_'+data_num+'.msg'), 'rb') as f:
    rscanmsg.deserialize(f.read())
lposemsg = PoseStamped()
with open(os.path.join(data_dir, 'self_pose_'+data_num+'.msg'), 'rb') as f:
    lposemsg.deserialize(f.read())
rposemsg = PoseStamped()
with open(os.path.join(data_dir, 'other_pose_'+data_num+'.msg'), 'rb') as f:
    rposemsg.deserialize(f.read())
with open(os.path.join(data_dir, 'meta_'+data_num+'.txt'), 'rb') as f:
    metadata = yaml.load(f.read())
text = '"{}"'.format(metadata['text'])

spectrogram = Trainer(channels).spectrogram([sound])[0]
plt.figure(figsize=(16,12),dpi=150)
for i in range(channels):
    plt.subplot(2,channels,i+1)
    spec = np.abs(spectrogram[:,:,i].T)
    spec /= spec.max()
    plt.imshow(spec, origin='lower')

player = SoundPlayer(channels=channels, sample_rate=rate)
player.play(sound)

_mapimg = np.array(mapmsg.data).reshape(mapmsg.info.height, mapmsg.info.width)
mapimg = np.ones([mapmsg.info.height,mapmsg.info.width,3], dtype=np.uint8) * 196
mapimg[np.where(_mapimg>0)] = 0

q = mapmsg.info.origin.orientation
rot = transformations.quaternion_matrix([q.x,q.y,q.z,q.w])
p = mapmsg.info.origin.position
trans = transformations.translation_matrix([p.x,p.y,p.z])
scale = transformations.scale_matrix(mapmsg.info.resolution)

#mapimg = cv2.resize(mapimg, (mapimg.shape[1]*4, mapimg.shape[0]*4))
#scale /= 4

pixel2map = np.matmul(trans, np.matmul(rot, scale))
map2pixel = np.linalg.inv(pixel2map)

p = lposemsg.pose.position
lx,ly,lz,_ = np.matmul(map2pixel, np.array([p.x, p.y, p.z, 1.]))
o = lposemsg.pose.orientation
o = transformations.quaternion_matrix([o.x, o.y, o.z, o.w])[:,0]
ldx, ldy = np.matmul(map2pixel, o)[:2]*0.2
p = rposemsg.pose.position
rx,ry,rz,_ = np.matmul(map2pixel, np.array([p.x, p.y, p.z, 1.]))
o = rposemsg.pose.orientation
o = transformations.quaternion_matrix([o.x, o.y, o.z, o.w])[:,0]
rdx, rdy = np.matmul(map2pixel, o)[:2]*0.2
cx, cy = (lx+rx)/2, (ly+ry)/2
xmin, xmax = cx-5/mapmsg.info.resolution, cx+5/mapmsg.info.resolution
ymin, ymax = cy-5/mapmsg.info.resolution, cy+5/mapmsg.info.resolution

lscanpoints = np.matmul(map2pixel, scan2points(lscanmsg, lposemsg.pose))
rscanpoints = np.matmul(map2pixel, scan2points(rscanmsg, rposemsg.pose))

plt.subplot(2,1,2)
plt.imshow(mapimg, origin='lower')
plt.plot(lx, ly, 'ro')
plt.plot(rx, ry, 'bo')
plt.scatter(lscanpoints[0], lscanpoints[1], s=1, c='r', linewidth=0)
plt.scatter(rscanpoints[0], rscanpoints[1], s=1, c='b', linewidth=0)
plt.arrow(x=lx,y=ly,dx=ldx,dy=ldy,width=.01/mapmsg.info.resolution,color='r')
plt.arrow(x=rx,y=ry,dx=rdx,dy=rdy,width=.01/mapmsg.info.resolution,color='b')
plt.text(rx, ry, text)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()
