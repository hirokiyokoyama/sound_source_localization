#!/usr/bin/env python

import rospy
import numpy as np
from sound_source_localization import SoundSourceLocalizer, SoundListener, QueueBuffer
from audio_common_msgs.msg import AudioData
from nav_msgs.msg import OccupancyGrid

rospy.init_node('ssl_predict')

ckpt = rospy.get_param('~ckpt')
channels = rospy.get_param('ssl/channels', 4)
sample_rate = rospy.get_param('ssl/sample_rate', 16000)
frame_length = rospy.get_param('ssl/stft_length', 480)
resolution = rospy.get_param('ssl/resolution', 0.5)
localize_rate = rospy.get_param('~rate', 5)

ssl = SoundSourceLocalizer(channels)
buf = QueueBuffer(frame_length, channels)
sl = SoundListener(buf)
ssl.restore_variables(ckpt)
pub = rospy.Publisher('sound_source_map', OccupancyGrid, queue_size=1)

rate = rospy.Rate(localize_rate)
while not rospy.is_shutdown():
    rate.sleep()
    #if pub.get_num_connections() == 0:
    #    continue

    data = buf.get_latest(frame_length)
    if data.shape[0] < frame_length:
        continue
    ssmap = ssl.sound_source_map(data, frame_step=frame_length)[0]

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(ssmap, origin='lower')
    plt.pause(0.01)

    grid = OccupancyGrid()
    grid.header.stamp = rospy.Time.now()
    grid.header.frame_id = 'base_footprint'
    grid.info.map_load_time = rospy.Time.now()
    grid.info.resolution = resolution
    grid.info.width = ssmap.shape[1]
    grid.info.height = ssmap.shape[0]
    grid.info.origin.position.x = -ssmap.shape[1]*resolution/2.
    grid.info.origin.position.y = -ssmap.shape[0]*resolution/2.
    grid.info.origin.orientation.w = 1.
    grid.data = np.int8(ssmap.reshape(-1)*100)
    pub.publish(grid)
