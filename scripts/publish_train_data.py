#!/usr/bin/env python

import rospy
import os
import numpy as np
from audio_common_msgs.msg import AudioData
from visualization_msgs.msg import MarkerArray, Marker
from dataset import get_recorded_dataset
from train import sound_source_gen, sound_gen
from train import NUM_DECONV
MAP_SIZE = 3*2**NUM_DECONV
MSG_FREQ = 100

if __name__=='__main__':
    rospy.init_node('ssl_train_data_pub')
    sample_rate = rospy.get_param('ssl/sample_rate')
    frame_length = rospy.get_param('ssl/stft_length')
    resolution = rospy.get_param('ssl/resolution')
    dataset_dir = rospy.get_param('~dataset_dir')

    files = os.listdir(os.path.abspath(dataset_dir))
    if all('sound' not in f for f in files):
        dataset_dir = [os.path.join(dataset_dir, f) for f in files]
        dataset_dir = filter(os.path.isdir, dataset_dir)
    dataset = get_recorded_dataset(dataset_dir)
    gen = sound_gen(sound_source_gen(dataset, MAP_SIZE, resolution, frame_length), batch_size=1)

    audio_pub = rospy.Publisher('train_sound', AudioData, queue_size=100)
    marker_pub = rospy.Publisher('train_ground_truth', MarkerArray, queue_size=1)
    assert sample_rate % MSG_FREQ == 0
    frames_per_msg = sample_rate / MSG_FREQ
    
    for sounds, positions in gen:
        if rospy.is_shutdown():
            break
        sounds = sounds[0]
        positions = positions[0]
        
        rate = rospy.Rate(MSG_FREQ)
        sound = np.int16(sounds.mean(0) * 32768)

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.ns = 'sound_sources'
        delete_all = MarkerArray(markers=[delete_all])
        
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
                marker.pose.position.x = (pos[0] - MAP_SIZE/2. + .5) * resolution
                marker.pose.position.y = (pos[1] - MAP_SIZE/2. + .5) * resolution
                marker.pose.orientation.w = 1.
                marker.scale.x = scale*5 * resolution
                marker.scale.y = scale*5 * resolution
                marker.scale.z = scale*5 * resolution
                marker.color.g = 1.
                marker.color.a = .5
                markers.markers.append(marker)
            marker_pub.publish(markers)
            
            rate.sleep()
            marker_pub.publish(delete_all)
