#!/usr/bin/env python

import rospy
import numpy as np
from sound_source_localization import SoundSourceLocalizer, SoundListener, QueueBuffer
from audio_common_msgs.msg import AudioData

rospy.init_node('ssl_predict')

ckpt = rospy.get_param('~ckpt')
channels = rospy.get_param('ssl/channels')
sample_rate = rospy.get_param('ssl/sample_rate')
frame_length = rospy.get_param('ssl/stft_length', 480)
localize_rate = rospy.get_param('~rate', 5)

ssl = SoundSourceLocalizer(channels)
buf = QueueBuffer(frame_length, channels)
sl = SoundListener(buf)
ssl.restore_variables(ckpt)

rate = rospy.Rate(localize_rate)
while not rospy.is_shutdown():
    rate.sleep()
    data = buf.get_latest(frame_length)
    ssmap = ssl.sound_source_map(data, frame_step=frame_length)[0]
    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(ssmap)
    plt.pause(0.01)
