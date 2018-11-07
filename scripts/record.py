#!/usr/bin/env python

import rospy
import numpy as np
from dataset import get_speech_commands_dataset
from utils import Synchronizer
from sound import SoundListener, SoundPlayer
from scipy.io import wavfile
import os
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import hsrb_interface
import yaml

#rospy.init_node('ssl_record')
robot = hsrb_interface.Robot()
tf_buffer = robot._get_tf2_buffer()
omni_base = robot.get('omni_base')
whole_body = robot.get('whole_body')

sound_dir = rospy.get_param('~sound_dir')
save_dir = rospy.get_param('~save_dir')
channels = rospy.get_param('ssl/channels')
sample_rate = rospy.get_param('ssl/sample_rate')
role_name = rospy.get_param('~role_name')
prefix = rospy.get_param('ssl/remote_prefix')
turn_angle = rospy.get_param('~turn_angle', 36)
turn_num = rospy.get_param('~turn_num', 10)
lift_min = rospy.get_param('~lift_min', 0.)
lift_max = rospy.get_param('~lift_max', .65)
lift_num = rospy.get_param('~lift_num', 10)

sound_dataset = get_speech_commands_dataset(sound_dir)
background_noises = sound_dataset.pop('_background_noise_')
sl = SoundListener(sample_rate*10, channels)
sp = SoundPlayer(channels=1, sample_rate=16000) # Sampling rate is 16kHz in speech_commands dataset
sync = Synchronizer()

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def tf2pose(tf):
    ps = PoseStamped()
    ps.header = tf.header
    ps.pose.position.x = tf.transform.translation.x
    ps.pose.position.y = tf.transform.translation.y
    ps.pose.position.z = tf.transform.translation.z
    ps.pose.orientation = tf.transform.rotation
    return ps

def serialize_rosmsg(msg):
    import StringIO
    sio = StringIO.StringIO()
    msg.serialize(sio)
    ret = sio.getvalue()
    sio.close()
    return ret

def phase_generator(phases):
    i = 0
    while True:
        for j, phase in enumerate(phases):
            yield i, j, phase
        i += 1
plan = []
for a, b in [('A', 'B'), ('B', 'A')]:
    plan += [{a: 'FACE', b: 'FACE'}]
    for _ in range(turn_num):
        for _ in range(lift_num):
            plan += [{a: 'LISTEN', b: 'SPEAK'},
                     {a: 'SAVE',   b: 'HIGHER'}]
        plan += [{a: 'TURN', b: 'LOWEST'}]
gen = phase_generator(plan)

mapmsg = rospy.wait_for_message('/static_distance_map_ref', OccupancyGrid)
with open(os.path.join(save_dir, 'map.msg'), 'wb') as f:
    mapmsg.serialize(f)
    
whole_body.move_to_neutral()
next_data = {}
while not rospy.is_shutdown():
    data = sync.next(next_data)
    next_data = {}
    iteration, phase, roles = gen.next()
    print 'Iteration %d, phase %d' % (iteration, phase)
    role = roles[role_name]
    print role
    if role == 'FACE':
        self_pose = rospy.wait_for_message('/laser_2d_pose', PoseWithCovarianceStamped).pose.pose
        other_pose = rospy.wait_for_message(prefix+'/laser_2d_pose', PoseWithCovarianceStamped).pose.pose
        #from tf.transformations import quaternion_inverse, quaternion_multiply
        #q = quaternion_multiply(other_pose.orientation, quaternion_inverse(self_pose.orientation))
        #angle = np.arctan2(q.z, q.w)*2
        self_ori = np.arctan2(self_pose.orientation.z, self_pose.orientation.w)*2
        other_dir = np.arctan2(other_pose.position.y-self_pose.position.y,
                               other_pose.position.x-self_pose.position.x)
        omni_base.go_rel(0, 0, other_dir-self_ori)
        whole_body.move_to_neutral()
    elif role == 'LISTEN':
        sl.start()
    elif role == 'SPEAK':
        key = np.random.choice(sound_dataset.keys()).tostring()
        filename = np.random.choice(sound_dataset[key]).tostring()
        rate, sound = wavfile.read(filename)
        assert rate == 16000
        sp.play(sound)
        speaker_tf = tf_buffer.lookup_transform('map', 'hand_palm_link', rospy.Time(0))
        speaker_pose = tf2pose(speaker_tf)
        base_tf = tf_buffer.lookup_transform('map', 'base_range_sensor_link', rospy.Time(0))
        base_pose = tf2pose(base_tf)
        rospy.sleep(1.)
        next_data = {'sound_file': filename,
                     'text': key,
                     'speaker_pose': serialize_rosmsg(speaker_pose),
                     'base_pose': serialize_rosmsg(base_pose)}
    elif role == 'SAVE':
        sound = sl.stop()
        msgs = {}
        msgs['self_scan'] = rospy.wait_for_message('/hsrb/base_scan', LaserScan)
        msgs['other_scan'] = rospy.wait_for_message(prefix+'/hsrb/base_scan', LaserScan)
        
        mic_tf = tf_buffer.lookup_transform('map', 'head_rgbd_sensor_link', rospy.Time(0))
        msgs['mic_pose'] = tf2pose(mic_tf)
        base_tf = tf_buffer.lookup_transform('map', 'base_range_sensor_link', rospy.Time(0))
        msgs['self_pose'] = tf2pose(base_tf)
        speaker_pose = PoseStamped()
        speaker_pose.deserialize(data.pop('speaker_pose'))
        msgs['speaker_pose'] = speaker_pose
        other_pose = PoseStamped()
        other_pose.deserialize(data.pop('base_pose'))
        msgs['other_pose'] = other_pose
        
        filename = os.path.join(save_dir, 'sound_{:04d}_{:04d}.wav'.format(iteration,phase))
        wavfile.write(filename, sample_rate, sound)
        for k, v in msgs.iteritems():
            filename = os.path.join(save_dir, '{}_{:04d}_{:04d}.msg'.format(k,iteration,phase))
            with open(filename, 'wb') as f:
                v.serialize(f)
        filename = os.path.join(save_dir, 'meta_{:04d}_{:04}.txt'.format(iteration,phase))        
        with open(filename, 'w') as f:
            f.write(yaml.dump(data, default_flow_style=False))
    elif role == 'TURN':
        omni_base.go_rel(0, 0, turn_angle/180.*np.pi)
        whole_body.move_to_neutral()
    elif role == 'HIGHER':
        current_pos = whole_body.joint_positions['arm_lift_joint']
        whole_body.move_to_joint_positions({'arm_lift_joint': current_pos+(lift_max-lift_min)/float(lift_num)})
    elif role == 'LOWEST':
        whole_body.move_to_neutral()
        whole_body.move_to_joint_positions({'arm_lift_joint': lift_min})
