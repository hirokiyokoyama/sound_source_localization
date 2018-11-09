import tensorflow as tf
slim = tf.contrib.slim

def conv_deconv(x, num_deconv=2, is_training=True):
    shape = tf.shape(x)
    N = shape[0]
    T = shape[1]
    if x.get_shape() is None or x.get_shape()[2].value is None:
        raise ValueError('The 3rd dimension (frequency) of inputs must be specified.')
    
    batch_norm_params = {'decay': 0.0005, 'epsilon': 0.00001,
                         'center': True, 'scale': True}
    with slim.arg_scope([slim.conv2d], stride=[1,2], padding='SAME',
                        normalizer_fn=slim.batch_norm):
     with slim.arg_scope([slim.conv2d_transpose], stride=2, padding='SAME',
                         normalizer_fn=slim.batch_norm):
      with slim.arg_scope([slim.batch_norm], is_training=is_training, **batch_norm_params):
       with slim.arg_scope([slim.max_pool2d], stride=2):
        #[N,T,F,C]
        net = slim.conv2d(x, 12, [3,6])
        net = slim.conv2d(net, 18, [3,6])
        net = slim.conv2d(net, 26, [3,6])
        net = slim.conv2d(net, 36, [3,6])

        F = net.get_shape()[2].value
        hidden = tf.reshape(net, [N,T,F*36])
        net = tf.reshape(net, [N*T,F,9,4])
        #[N*T,F,9*c]
        net = tf.transpose(net, [0,2,1,3])
        #[N*T,9,F,c]
        net = tf.reshape(net, [N*T,3,3,F*4])
        #[N*T,3,3,F*c]

        net = slim.conv2d(net, 2**num_deconv, [1,1], stride=1)
        #[N*T,3,3,2**n]
        for i in range(num_deconv):
            net = slim.conv2d_transpose(net, 2**(num_deconv-i-1), [3,3])
        #[N*T,3*2**n,3*2**n,1]
        W = 3*2**num_deconv
        net = tf.reshape(net, [N,T,W,W])
    return hidden, net
