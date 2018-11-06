#!/usr/bin/env python

import rospy
import numpy as np
from dataset import get_speech_commands_dataset
from utils import Synchronizer
from sound import SoundListener, SoundPlayer
from scipy.io import wavfile
import os
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import hsrb_interface

#rospy.init_node('ssl_record')
robot = hsrb_interface.Robot()
omni_base = robot.get('omni_base')

sound_dir = rospy.get_param('~sound_dir')
save_dir = rospy.get_param('~save_dir')
channels = rospy.get_param('ssl/channels')
sample_rate = rospy.get_param('ssl/sample_rate')
role_name = rospy.get_param('~role_name')
prefix = rospy.get_param('ssl/remote_prefix')
turn_angle = rospy.get_param('~turn_angle', 36)
turn_num = rospy.get_param('~turn_num', 10)
speak_num = rospy.get_param('~speak_num', 10)

sound_dataset = get_speech_commands_dataset(sound_dir)
background_noises = sound_dataset.pop('_background_noise_')
sl = SoundListener(sample_rate*10, channels)
sp = SoundPlayer(channels=1, sample_rate=16000) # Sampling rate is 16kHz in speech_commands dataset
sync = Synchronizer()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def phase_generator(phases):
    i = 0
    while True:
        for j, phase in enumerate(phases):
            yield i, j, phase
        i += 1
plan = []
for a, b in [('A', 'B'), ('B', 'A')]:
    for _ in range(turn_num):
        for _ in range(speak_num):
            plan += [{a: 'LISTEN', b: 'SPEAK'},
                     {a: 'SAVE',   b: 'WAIT'}]
        plan += [{a: 'TURN', b: 'WAIT'}]
gen = phase_generator(plan)

mapmsg = rospy.wait_for_message('/static_distance_map_ref', OccupancyGrid)
with open(os.path.join(save_dir, 'map.msg'), 'wb') as f:
    mapmsg.serialize(f)

while not rospy.is_shutdown():
    sync.next()
    iteration, phase, roles = gen.next()
    print 'Iteration %d, phase %d' % (iteration, phase)
    role = roles[role_name]
    print role
    if role == 'LISTEN':
        sl.start()
    elif role == 'SPEAK':
        key = np.random.choice(sound_dataset.keys())
        filename = np.random.choice(sound_dataset[key])
        rate, data = wavfile.read(filename)
        assert rate == 16000
        sp.play(data)
        rospy.sleep(1.)
    elif role == 'SAVE':
        data = sl.stop()
        local_scan = rospy.wait_for_message('/hsrb/base_scan', LaserScan)
        remote_scan = rospy.wait_for_message(prefix+'/hsrb/base_scan', LaserScan)
        local_pose = rospy.wait_for_message('/laser_2d_pose', PoseWithCovarianceStamped)
        remote_pose = rospy.wait_for_message(prefix+'/laser_2d_pose', PoseWithCovarianceStamped)
        
        filename = os.path.join(save_dir, 'sound_{:04d}_{:04d}.wav'.format(iteration,phase))
        wavfile.write(filename, sample_rate, data)
        filename = os.path.join(save_dir, 'local_pose_{:04d}_{:04d}.msg'.format(iteration,phase))
        with open(filename, 'wb') as f:
            local_pose.serialize(f)
        filename = os.path.join(save_dir, 'remote_pose_{:04d}_{:04d}.msg'.format(iteration,phase))
        with open(filename, 'wb') as f:
            remote_pose.serialize(f)
    elif role == 'TURN':
        omni_base.go_rel(0, 0, turn_angle/180.*np.pi)
