#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from dataset import get_recorded_dataset
from sound_source_localization import SoundSourceLocalizer, SoundMatcher, axis_vector
import os
from concurrent import futures

FRAME_STEP = 160
NUM_DECONV = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 8
MAX_SOURCES = 3

def sound_source_gen(dataset, W, resolution, frame_length, threshold=0.7):
    matcher = SoundMatcher(frame_length, FRAME_STEP)
    inds = range(len(dataset))
    while True:
        np.random.shuffle(inds)
        for i in inds:
            data = dataset[i]

            rate, sound = wavfile.read(data['recorded_sound_file'])
            sound = sound / 32768.

            if 'span' in data:
                begin, end = data['span']
                confidence = data['confidence']
                pos = data['source_position']
            else:
                _rate, _sound = wavfile.read(data['sound_file'])
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
                x = W/2 + int(np.round(x/resolution))
                y = W/2 + int(np.round(y/resolution))
                x = min(max(x, 0), W-1)
                y = min(max(y, 0), W-1)
                pos = (x,y)
                data['source_position'] = pos

            if threshold is None or confidence > threshold:
                yield sound[begin:end,:], pos

def _process(data, max_sources, length, begins):
    sounds = np.zeros([max_sources, length, 4], dtype=np.float32)
    pos = np.zeros([max_sources, 2], dtype=np.int32)

    for i, (s, p) in enumerate(data):
        begin = begins[i]
        end = begin + s.shape[0]
        sounds[i,begin:end,:] = s
        pos[i] = p
    return sounds, pos

class DummyExecutor:
    def submit(self, func, *args, **kwargs):
        class Future:
            def __init__(self, r):
                self._result = r
            def result(self):
                return self._result
        return Future(func(*args, **kwargs))
    
def sound_gen(gen, batch_size=8, max_sources=3):
    executor = futures.ProcessPoolExecutor()
    while True:
        ns = [np.random.randint(1,max_sources+1) for _ in range(batch_size)]
        data = [gen.next() for _ in range(np.prod(ns))]
        T = max(s.shape[0] for s, p in data)
        
        future_data = []
        for b, n in enumerate(ns):
            _data = data[:n]
            data = data[n:]
            begins = [np.random.randint(T*2 - s.shape[0]) for s, p in _data]
            future_data.append(executor.submit(_process, _data, max_sources, T*2, begins))
        sounds = []
        pos = []
        for f in future_data:
            s, p = f.result()
            sounds.append(s)
            pos.append(p)
        sounds = np.stack(sounds, 0)
        pos = np.stack(pos, 0)
        yield sounds, pos

if __name__=='__main__':
    import tensorflow as tf
    import rospy

    rospy.init_node('ssl_train', disable_signals=True) # just to get parameters
    channels = rospy.get_param('ssl/channels')
    #sample_rate = rospy.get_param('ssl/sample_rate')
    frame_length = rospy.get_param('ssl/stft_length')
    resolution = rospy.get_param('ssl/resolution')
    model_dir = rospy.get_param('~model_dir')
    dataset_dir = rospy.get_param('~dataset_dir')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    trainer = SoundSourceLocalizer(channels, frame_length, FRAME_STEP, NUM_DECONV,
                                   learning_rate = LEARNING_RATE)
    
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt is not None:
        print 'Restoring from %s.' % latest_ckpt
        trainer.restore_variables(latest_ckpt)
    else:
        print 'Starting with a new model.'
        trainer.initialize_variables()

    files = os.listdir(os.path.abspath(dataset_dir))
    if all('sound' not in f for f in files):
        dataset_dir = [os.path.join(dataset_dir, f) for f in files]
        dataset_dir = filter(os.path.isdir, dataset_dir)
    dataset = get_recorded_dataset(dataset_dir)
    gen = sound_gen(sound_source_gen(dataset, trainer.map_size, resolution, frame_length),
                    batch_size=BATCH_SIZE, max_sources=MAX_SOURCES)

    for sounds, positions in gen:
        if rospy.is_shutdown():
            break
        step, losses = trainer.train(sounds, positions)
        print 'loss', losses.mean()
        
        #from sound import SoundPlayer
        #SoundPlayer().play(np.int16(sounds[:,:,0:1].mean(0)*32768))
        #trainer.plot(sounds, positions)
        if step % 1000 == 0:
            trainer.save_variables(os.path.join(model_dir, 'model'))
