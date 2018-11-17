#!/usr/bin/env python

import rospy
import sys
import numpy as np
from audio_common_msgs.msg import AudioData
from visualization_msgs.msg import MarkerArray, Marker
from dataset import get_recorded_dataset
from train import sound_source_gen, sound_gen
from train import SAMPLE_RATE, NUM_DECONV, RESOLUTION
MAP_SIZE = 3*2**NUM_DECONV
MSG_FREQ = 100

if __name__=='__main__':
    rospy.init_node('ssl_train_data_pub')
    dataset_dir = rospy.get_param('~dataset_dir')
    dataset = get_recorded_dataset(dataset_dir)
    gen = sound_gen(sound_source_gen(dataset, MAP_SIZE), batch_size=1)

    audio_pub = rospy.Publisher('train_sound', AudioData, queue_size=100)
    marker_pub = rospy.Publisher('train_ground_truth', MarkerArray, queue_size=1)
    assert SAMPLE_RATE % MSG_FREQ == 0
    frames_per_msg = SAMPLE_RATE / MSG_FREQ
    
    for sounds, positions in gen:
        if rospy.is_shutdown():
            break
        sounds = sounds[0]
        positions = positions[0]
        
        rate = rospy.Rate(MSG_FREQ)
        sound = np.int16(sounds.mean(0) * 32768)
        for t in xrange((sound.shape[0]+frames_per_msg-1)/frames_per_msg):
            begin = t*frames_per_msg
            end = min((t+1)*frames_per_msg, sound.shape[0])
            audio_pub.publish(AudioData(data = sound[begin:end].tostring()))

            markers = MarkerArray()
            for i, (snd, pos) in enumerate(zip(sounds, positions)):
                scale = np.abs(snd[begin:end]).max()
                if scale == 0.:
                    continue
                marker = Marker()
                marker.header.stamp = rospy.Time.now()
                marker.header.frame_id = 'base_footprint'
                marker.type = Marker.SPHERE
                marker.ns = 'sound_sources'
                marker.id = i
                marker.pose.position.x = (pos[0] - MAP_SIZE/2. + .5) * RESOLUTION
                marker.pose.position.y = (pos[1] - MAP_SIZE/2. + .5) * RESOLUTION
                marker.pose.orientation.w = 1.
                marker.scale.x = scale*5 * RESOLUTION
                marker.scale.y = scale*5 * RESOLUTION
                marker.scale.z = scale*5 * RESOLUTION
                marker.color.g = 1.
                marker.color.a = .5
                markers.markers.append(marker)
            marker_pub.publish(markers)
            
            rate.sleep()
