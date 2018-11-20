import tensorflow as tf
import numpy as np
from nets import conv_deconv

class SoundMatcher:
    def __init__(self, frame_length=480, frame_step=160, session_config=None):
        self._frame_step = frame_step
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._sound = tf.placeholder(tf.float32, shape=[None,None])
            self._pattern = tf.placeholder(tf.float32, shape=[None,None])

            sound = tf.transpose(self._sound, [1,0])
            sound = tf.contrib.signal.stft(sound, frame_length, frame_step,
                                           pad_end = True)
            sound = tf.reduce_mean(tf.abs(sound), 0)
            self._sound_spectrogram = sound
            sound = tf.expand_dims(sound, -1)
            pattern = tf.transpose(self._pattern, [1,0])
            pattern = tf.contrib.signal.stft(pattern, frame_length, frame_step,
                                             pad_end = True)
            pattern = tf.reduce_mean(tf.abs(pattern), 0)
            self._pattern_spectrogram = pattern
            pattern = tf.expand_dims(pattern, -1)
            shape = tf.shape(pattern)
            corr = tf.nn.convolution(tf.expand_dims(sound,0), tf.expand_dims(pattern,-1),
                                     padding = 'VALID')[0,:,0,0]
            a = tf.nn.convolution(tf.expand_dims(sound,0)**2, tf.ones([shape[0],shape[1],1,1]),
                                  padding = 'VALID')[0,:,0,0]
            b = tf.reduce_sum(pattern**2)
            self._correlation = corr/tf.sqrt(a*b)
        self._sess = tf.Session(graph=self._graph, config=session_config)

    def match(self, sound, pattern):
        corr = self._sess.run(self._correlation,
                              {self._sound: sound, self._pattern: pattern})
        begin = corr.argmax()
        val = corr[begin]
        begin *= self._frame_step
        end = begin + pattern.shape[0]
        return begin, end, val

    def plot(self, sound, pattern):
        _sound, _pattern, corr = self._sess.run([self._sound_spectrogram, self._pattern_spectrogram, self._correlation],
                                                {self._sound: sound, self._pattern: pattern})
        begin = corr.argmax()
        end = begin + pattern.shape[0]/self._frame_step
        
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        ax = plt.subplot(2,2,1)
        plt.imshow(_sound.T, origin='lower')
        ax.add_line(Line2D([begin,begin], [0, _sound.shape[1]], linewidth=2, color='r'))
        ax.add_line(Line2D([end,end], [0, _sound.shape[1]], linewidth=2, color='r'))
        plt.subplot(2,2,2)
        plt.imshow(_pattern.T, origin='lower')
        plt.subplot(2,1,2)
        plt.plot(corr)

class SoundSourceLocalizer:
    def __init__(self, channels, frame_length=480, frame_step=160, num_deconv=2,
                 learning_rate = 0.001, session_config = None):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._W = 3*2**num_deconv
            self._global_step = tf.train.create_global_step()
            self._sound_sources = tf.placeholder(tf.float32, shape=[None,None,None,channels])
            self._positions = tf.placeholder(tf.int32, shape=[None,None,2])
            self._frame_step = tf.placeholder_with_default(frame_step, shape=[])
            self._is_training = tf.placeholder_with_default(False, shape=[])
            sound_sources = tf.transpose(self._sound_sources, [0,1,3,2])
            spectrogram = tf.contrib.signal.stft(sound_sources,
                                                 frame_length,
                                                 self._frame_step,
                                                 pad_end=True)
            # [M,N,C,T,F]
            shape = tf.shape(spectrogram)
            T = shape[3]
            M = shape[0]
            envelopes = tf.reduce_mean(tf.reduce_mean(tf.abs(spectrogram), 4), 2)
            cond = lambda i, labels: i < M
            def body(i, labels):
                next_labels = tf.scatter_nd(self._positions[i], envelopes[i], [self._W,self._W,T])
                next_labels = tf.expand_dims(next_labels, 0)
                labels = tf.concat([labels, next_labels], 0)
                return i+1, labels
            _, self._labels = tf.while_loop(cond, body,
                                            loop_vars=[0, tf.zeros([0,self._W,self._W,T])],
                                            shape_invariants=[tf.TensorShape([]), tf.TensorShape([None,self._W,self._W,None])])
            self._labels = tf.transpose(self._labels, [0,3,1,2])
            # [M,T,W,W]
            self._spectrogram = tf.transpose(tf.reduce_sum(spectrogram, 1), [0,2,3,1])
            # [M,T,F,C]
            self._features = tf.concat([tf.real(self._spectrogram), tf.imag(self._spectrogram)], -1)
            self._logits = conv_deconv(self._features, num_deconv=num_deconv, is_training=self._is_training)
            self._sound_source_map = tf.nn.sigmoid(self._logits)
            self._losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self._logits, labels=self._labels)
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self._train_op = opt.minimize(tf.reduce_mean(self._losses), global_step=self._global_step)
            
            self._init_op = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._sess = tf.Session(graph=self._graph, config=session_config)

    @property
    def map_size(self):
        return self._W

    def initialize_variables(self):
        self._sess.run(self._init_op)

    def restore_variables(self, ckpt):
        self._saver.restore(self._sess, ckpt)

    def spectrogram(self, sound, frame_step=None):
        feed_dict = {self._sound_sources: [[sound]]}
        if frame_step is not None:
            feed_dict[self._frame_step] = frame_step
        return self._sess.run(self._spectrogram, feed_dict)[0]

    def sound_source_map(self, sound, frame_step=None):
        feed_dict = {self._sound_sources: [[sound]]}
        if frame_step is not None:
            feed_dict[self._frame_step] = frame_step
        return self._sess.run(self._sound_source_map, feed_dict)[0]

    def train(self, sound_sources, positions, frame_step=None, return_prediction=False):
        feed_dict = {self._sound_sources: sound_sources,
                     self._positions: positions,
                     self._is_training: True}
        if frame_step is not None:
            feed_dict[self._frame_step] = frame_step
        fetch_list = [self._train_op, self._global_step, self._losses]
        if return_prediction:
            fetch_list.append(self._sound_source_map)
            _, step, losses, pred = self._sess.run(fetch_list, feed_dict)
            return step, losses, pred
        _, step, losses = self._sess.run(fetch_list, feed_dict)
        return step, losses

    def save_variables(self, ckpt):
        self._saver.save(self._sess, ckpt, global_step=self._global_step)
        
    def plot(self, sound_sources, positions):
        spectrogram, labels = self._sess.run([self._spectrogram, self._labels],
                                             {self._sound_sources: [sound_sources],
                                              self._positions: [positions]})
        spectrogram = np.abs(spectrogram[0])
        labels = labels[0]
        
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.lines import Line2D
        
        fig = plt.figure('sound', figsize=(40,20))
        plt.subplot(1,2,1)
        labels_im = plt.imshow(labels[0], origin='lower', cmap='Reds',
                               vmin=0., vmax=labels.max())
        plt.colorbar()
        time_lines = []
        for i in range(spectrogram.shape[2]):
            ax = plt.subplot(spectrogram.shape[2],2,i*2+2)
            plt.imshow(spectrogram[:,:,i].T, origin='lower', aspect='auto')
            time_lines.append(ax.add_line(Line2D([0,0], [0,spectrogram.shape[1]], linewidth=1, color='r')))
        
        def update(frame):
            for time_line in time_lines:
                time_line.set_data([frame,frame], [0,spectrogram.shape[1]])
            labels_im.set_data(labels[frame])
            return time_lines+[labels_im]
        a = animation.FuncAnimation(fig, update, frames=spectrogram.shape[0], interval=1)
        plt.show()
