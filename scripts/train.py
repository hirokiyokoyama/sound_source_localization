#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from dataset import get_recorded_dataset
from sound_source_localization import SoundSourceLocalizer, SoundMatcher, axis_vector
import os, sys

CHANNELS = 4
SAMPLE_RATE = 16000
BUFFER_SIZE = SAMPLE_RATE/5
FRAME_LENGTH = 480
FRAME_STEP = 160
NUM_DECONV = 2
RESOLUTION = 0.5
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models')
LEARNING_RATE = 0.001

def sound_source_gen(dataset, W):
    inds = range(len(dataset))
    while True:
        np.random.shuffle(inds)
        for i in inds:
            data = dataset[i]

            # ONLY FOR NOW!
            data['sound_file'] = data['sound_file'].replace('roboworks', 'yokoyama')
            rate, sound = wavfile.read(data['recorded_sound_file'])
            assert rate == SAMPLE_RATE
            assert sound.shape[1] == CHANNELS
            sound = sound / 32768.

            if 'span' in data:
                begin, end = data['span']
                confidence = data['confidence']
                pos = data['source_position']
            else:
                _rate, _sound = wavfile.read(data['sound_file'])
                assert _rate == SAMPLE_RATE
                if len(_sound.shape) == 1:
                    _sound = np.expand_dims(_sound, 1)
                _sound = _sound / 32768.
                begin, end, confidence = matcher.match(sound, _sound)
                data['span'] = (begin, end)
                data['confidence'] = confidence
                
                mic_pos = data['mic_pose'].pose.position
                mic_pos = np.array([mic_pos.x, mic_pos.y, mic_pos.z])
                mic_vec = axis_vector(data['mic_pose'].pose.orientation, 2)
                speaker_pos = data['speaker_pose'].pose.position
                speaker_pos = np.array([speaker_pos.x, speaker_pos.y, speaker_pos.z])
                speaker_vec = axis_vector(data['speaker_pose'].pose.orientation, 2)
                mic_ori = np.arctan2(mic_vec[1], mic_vec[0])
                d = speaker_pos-mic_pos
                dist = np.hypot(d[0], d[1])
                theta = np.arctan2(d[1], d[0]) - mic_ori
                x = dist * np.cos(theta)
                y = dist * np.sin(theta)
                x = W/2 + int(np.round(x/RESOLUTION))
                y = W/2 + int(np.round(y/RESOLUTION))
                x = min(max(x, 0), W-1)
                y = min(max(y, 0), W-1)
                pos = (x,y)
                data['source_position'] = pos

            yield sound[begin:end,:], pos

def sound_gen(gen, max_sources=3):
    while True:
        n = np.random.randint(1,max_sources+1)
        data = [gen.next() for _ in range(n)]
        T = max(s.shape[0] for s, p in data)
        
        sounds = np.zeros([n, T*2, 4], dtype=np.float32)
        pos = np.zeros([n, 2], dtype=np.int32)
        for i, (s, p) in enumerate(data):
            b = np.random.randint(sounds.shape[1] - s.shape[0])
            e = b + s.shape[0]
            sounds[i,b:e,:] = s
            pos[i] = p
        yield sounds, pos

if __name__=='__main__':
    print MODEL_DIR
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    trainer = SoundSourceLocalizer(CHANNELS, FRAME_LENGTH, FRAME_STEP, NUM_DECONV,
                                   learning_rate = LEARNING_RATE)
    matcher = SoundMatcher(FRAME_LENGTH, FRAME_STEP)
    
    latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
    if latest_ckpt is not None:
        print 'Restoring from %s.' % latest_ckpt
        trainer.restore_variables(latest_ckpt)
    else:
        print 'Starting with a new model.'
        trainer.initialize_variables()

    dataset = get_recorded_dataset(sys.argv[1:])
    gen = sound_gen(sound_source_gen(dataset, trainer.map_size))
    for sounds, positions in gen:
        step, losses = trainer.train(sounds, positions)
        print losses.mean()
        #from sound import SoundPlayer
        #SoundPlayer().play(np.int16(sounds[:,:,0:1].mean(0)*32768))
        #trainer.plot(sounds, positions)
        if step % 10 == 0:
            trainer.save_variables(os.path.join(MODEL_DIR, 'model'))
