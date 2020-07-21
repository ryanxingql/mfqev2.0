import tensorflow as tf
import tensorflow.contrib.slim as slim
import tflearn

def network(frame1, frame2, frame3, is_training, reuse=False, scope='netflow'): # design for QP37,42

    with tf.variable_scope(scope, reuse=reuse):

        # Define multi-scale feature extraction network

        c3_1_w = tf.get_variable("c3_1_w", shape=[3, 3, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c3_1_b = tf.get_variable("c3_1_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c3_2_w = tf.get_variable("c3_2_w", shape=[3, 3, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c3_2_b = tf.get_variable("c3_2_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c3_3_w = tf.get_variable("c3_3_w", shape=[3, 3, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c3_3_b = tf.get_variable("c3_3_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c5_1_w = tf.get_variable("c5_1_w", shape=[5, 5, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c5_1_b = tf.get_variable("c5_1_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c5_2_w = tf.get_variable("c5_2_w", shape=[5, 5, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c5_2_b = tf.get_variable("c5_2_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c5_3_w = tf.get_variable("c5_3_w", shape=[5, 5, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c5_3_b = tf.get_variable("c5_3_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c7_1_w = tf.get_variable("c7_1_w", shape=[7, 7, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c7_1_b = tf.get_variable("c7_1_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c7_2_w = tf.get_variable("c7_2_w", shape=[7, 7, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c7_2_b = tf.get_variable("c7_2_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c7_3_w = tf.get_variable("c7_3_w", shape=[7, 7, 1, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c7_3_b = tf.get_variable("c7_3_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        # Define dense reconstruction network

        c1_w = tf.get_variable("c1_w", shape=[3, 3, 32*3*3, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c1_b = tf.get_variable("c1_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c2_w = tf.get_variable("c2_w", shape=[3, 3, 32, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c2_b = tf.get_variable("c2_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c3_w = tf.get_variable("c3_w", shape=[3, 3, 32*2, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c3_b = tf.get_variable("c3_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c4_w = tf.get_variable("c4_w", shape=[3, 3, 32*3, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c4_b = tf.get_variable("c4_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c5_w = tf.get_variable("c5_w", shape=[3, 3, 32*4, 32],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c5_b = tf.get_variable("c5_b", shape=[32],
                               initializer=tf.constant_initializer(0.0))

        c6_w = tf.get_variable("c6_w", shape=[3, 3, 32, 1],
                               initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        c6_b = tf.get_variable("c6_b", shape=[1],
                               initializer=tf.constant_initializer(0.0))


        # Multi-scale feature extraction

        c3_1 = tf.nn.conv2d(frame1, c3_1_w, strides=[1, 1, 1, 1], padding='SAME')
        c3_1 = tf.nn.bias_add(c3_1, c3_1_b)
        c3_1 = tflearn.activations.prelu(c3_1)

        c5_1 = tf.nn.conv2d(frame1, c5_1_w, strides=[1, 1, 1, 1], padding='SAME')
        c5_1 = tf.nn.bias_add(c5_1, c5_1_b)
        c5_1 = tflearn.activations.prelu(c5_1)

        c7_1 = tf.nn.conv2d(frame1, c7_1_w, strides=[1, 1, 1, 1], padding='SAME')
        c7_1 = tf.nn.bias_add(c7_1, c7_1_b)
        c7_1 = tflearn.activations.prelu(c7_1)

        cc_1 = tf.concat([c3_1, c5_1, c7_1], 3)

        c3_2 = tf.nn.conv2d(frame2, c3_2_w, strides=[1, 1, 1, 1], padding='SAME')
        c3_2 = tf.nn.bias_add(c3_2, c3_2_b)
        c3_2 = tflearn.activations.prelu(c3_2)

        c5_2 = tf.nn.conv2d(frame2, c5_2_w, strides=[1, 1, 1, 1], padding='SAME')
        c5_2 = tf.nn.bias_add(c5_2, c5_2_b)
        c5_2 = tflearn.activations.prelu(c5_2)

        c7_2 = tf.nn.conv2d(frame2, c7_2_w, strides=[1, 1, 1, 1], padding='SAME')
        c7_2 = tf.nn.bias_add(c7_2, c7_2_b)
        c7_2 = tflearn.activations.prelu(c7_2)

        cc_2 = tf.concat([c3_2, c5_2, c7_2], 3)

        c3_3 = tf.nn.conv2d(frame3, c3_3_w, strides=[1, 1, 1, 1], padding='SAME')
        c3_3 = tf.nn.bias_add(c3_3, c3_3_b)
        c3_3 = tflearn.activations.prelu(c3_3)

        c5_3 = tf.nn.conv2d(frame3, c5_3_w, strides=[1, 1, 1, 1], padding='SAME')
        c5_3 = tf.nn.bias_add(c5_3, c5_3_b)
        c5_3 = tflearn.activations.prelu(c5_3)

        c7_3 = tf.nn.conv2d(frame3, c7_3_w, strides=[1, 1, 1, 1], padding='SAME')
        c7_3 = tf.nn.bias_add(c7_3, c7_3_b)
        c7_3 = tflearn.activations.prelu(c7_3)

        cc_3 = tf.concat([c3_3, c5_3, c7_3], 3)

        # Merge
        c_concat = tf.concat([cc_1, cc_2, cc_3], 3)

        # Dense + BN reconstruction

        c1 = tf.nn.conv2d(c_concat, c1_w, strides=[1, 1, 1, 1], padding='SAME')
        c1 = tf.nn.bias_add(c1, c1_b)
        c1 = tf.layers.batch_normalization(c1,training=is_training)
        c1 = tflearn.activations.prelu(c1)

        c2 = tf.nn.conv2d(c1, c2_w, strides=[1, 1, 1, 1], padding='SAME')
        c2 = tf.nn.bias_add(c2, c2_b)
        c2 = tf.layers.batch_normalization(c2,training=is_training)
        c2 = tflearn.activations.prelu(c2)

        cc2 = tf.concat([c1, c2], 3)

        c3 = tf.nn.conv2d(cc2, c3_w, strides=[1, 1, 1, 1], padding='SAME')
        c3 = tf.nn.bias_add(c3, c3_b)
        c3 = tf.layers.batch_normalization(c3,training=is_training)
        c3 = tflearn.activations.prelu(c3)

        cc3 = tf.concat([c1, c2, c3], 3)

        c4 = tf.nn.conv2d(cc3, c4_w, strides=[1, 1, 1, 1], padding='SAME')
        c4 = tf.nn.bias_add(c4, c4_b)
        c4 = tf.layers.batch_normalization(c4,training=is_training)
        c4 = tflearn.activations.prelu(c4)

        cc4 = tf.concat([c1, c2, c3, c4], 3)

        c5 = tf.nn.conv2d(cc4, c5_w, strides=[1, 1, 1, 1], padding='SAME')
        c5 = tf.nn.bias_add(c5, c5_b)
        c5 = tf.layers.batch_normalization(c5,training=is_training)
        c5 = tflearn.activations.prelu(c5)

        c6 = tf.nn.conv2d(c5, c6_w, strides=[1, 1, 1, 1], padding='SAME')
        c6 = tf.nn.bias_add(c6, c6_b)
        c6 = tf.layers.batch_normalization(c6,training=is_training)
        c6 = tflearn.activations.prelu(c6)

        # Short connection

        output = tf.add(c6, frame2)

        return output


def network2(frame1, frame2, frame3, reuse=False, scope='netflow'): # design for QP22, 27, 32

    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d],  activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            # Multi-scale

            c3_1 = slim.conv2d(frame1, 32, [3, 3], scope='conv3_1')
            c5_1 = slim.conv2d(frame1, 32, [5, 5], scope='conv5_1')
            c7_1 = slim.conv2d(frame1, 32, [7, 7], scope='conv7_1')

            cc_1 = tf.concat([c3_1, c5_1, c7_1], 3, name='concat_1')

            c3_2 = slim.conv2d(frame2, 32, [3, 3], scope='conv3_2')
            c5_2 = slim.conv2d(frame2, 32, [5, 5], scope='conv5_2')
            c7_2 = slim.conv2d(frame2, 32, [7, 7], scope='conv7_2')

            cc_2 = tf.concat([c3_2, c5_2, c7_2], 3, name='concat_2')

            c3_3 = slim.conv2d(frame3, 32, [3, 3], scope='conv3_3')
            c5_3 = slim.conv2d(frame3, 32, [5, 5], scope='conv5_3')
            c7_3 = slim.conv2d(frame3, 32, [7, 7], scope='conv7_3')

            cc_3 = tf.concat([c3_3, c5_3, c7_3], 3, name='concat_3')

            # Merge

            c_concat = tf.concat([cc_1, cc_2, cc_3], 3, name='c_concat')

            # General CNN

            cc1 = slim.conv2d(c_concat, 32, [3, 3], scope='cconv1')
            cc2 = slim.conv2d(cc1, 32, [3, 3], scope='cconv2')
            cc3 = slim.conv2d(cc2, 32, [3, 3], scope='cconv3')
            cc4 = slim.conv2d(cc3, 32, [3, 3], scope='cconv4')
            cc5 = slim.conv2d(cc4, 32, [3, 3], scope='cconv5')
            cc6 = slim.conv2d(cc5, 32, [3, 3], scope='cconv6')
            cc7 = slim.conv2d(cc6, 32, [3, 3], scope='cconv7')
            cc8 = slim.conv2d(cc7, 16, [3, 3], scope='cconv8')
            cout = slim.conv2d(cc8, 1, [3, 3], activation_fn=None, scope='cout') # 1 channel output

            output = tf.add(cout, frame2) # ResNet

        return output


def transformer(batch, chan, flow, U, out_size, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _repeat2(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1)
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(rep, tf.reshape(x, (1, -1)))
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            x = tf.cast(_repeat2(tf.range(0, width), height * num_batch), 'float32') + x * 64
            y = tf.cast(_repeat2(_repeat(tf.range(0, height), width), num_batch), 'float32') + y * 64

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)

            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(x_s, y_s, input_dim, out_size):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]

            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([batch, out_height, out_width, chan]))
            return output

    with tf.variable_scope(name):
        dx, dy = tf.split(flow, 2, 3)
        output = _transform(dx, dy, U, out_size)
        return output


def warp_img(batch_size, imga, imgb, reuse, scope='easyflow'):

    n, h, w, c = imga.get_shape().as_list()

    with tf.variable_scope(scope, reuse=reuse):

        with slim.arg_scope([slim.conv2d], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)), \
             slim.arg_scope([slim.conv2d_transpose], activation_fn=tflearn.activations.prelu,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):
            inputs = tf.concat([imga, imgb], 3, name='flow_inp')
            c1 = slim.conv2d(inputs, 24, [5, 5], stride=2, scope='c1')
            c2 = slim.conv2d(c1, 24, [3, 3], scope='c2')
            c3 = slim.conv2d(c2, 24, [5, 5], stride=2, scope='c3')
            c4 = slim.conv2d(c3, 24, [3, 3], scope='c4')
            c5 = slim.conv2d(c4, 32, [3, 3], activation_fn=tf.nn.tanh, scope='c5')
            c5_hr = tf.reshape(c5, [n, int(h / 4), int(w / 4), 2, 4, 4])
            c5_hr = tf.transpose(c5_hr, [0, 1, 4, 2, 5, 3])
            c5_hr = tf.reshape(c5_hr, [n, h, w, 2])
            img_warp1 = transformer(batch_size, c, c5_hr, imgb, [h, w])

            c5_pack = tf.concat([inputs, c5_hr, img_warp1], 3, name='cat')
            s1 = slim.conv2d(c5_pack, 24, [5, 5], stride=2, scope='s1')
            s2 = slim.conv2d(s1, 24, [3, 3], scope='s2')
            s3 = slim.conv2d(s2, 24, [3, 3], scope='s3')
            s4 = slim.conv2d(s3, 24, [3, 3], scope='s4')
            s5 = slim.conv2d(s4, 8, [3, 3], activation_fn=tf.nn.tanh, scope='s5')
            s5_hr = tf.reshape(s5, [n, int(h / 2), int(w / 2), 2, 2, 2])
            s5_hr = tf.transpose(s5_hr, [0, 1, 4, 2, 5, 3])
            s5_hr = tf.reshape(s5_hr, [n, h, w, 2])
            uv = c5_hr + s5_hr
            img_warp2 = transformer(batch_size, c, uv, imgb, [h, w])

            s5_pack = tf.concat([inputs, uv, img_warp2], 3, name='cat2')
            a1 = slim.conv2d(s5_pack, 24, [3, 3], scope='a1')
            a2 = slim.conv2d(a1, 24, [3, 3], scope='a2')
            a3 = slim.conv2d(a2, 24, [3, 3], scope='a3')
            a4 = slim.conv2d(a3, 24, [3, 3], scope='a4')
            a5 = slim.conv2d(a4, 2, [3, 3], activation_fn=tf.nn.tanh, scope='a5')
            a5_hr = tf.reshape(a5, [n, h, w, 2, 1, 1])
            a5_hr = tf.transpose(a5_hr, [0, 1, 4, 2, 5, 3])
            a5_hr = tf.reshape(a5_hr, [n, h, w, 2])
            uv2 = a5_hr + uv
            img_warp3 = transformer(batch_size, c, uv2, imgb, [h, w])

            tf.summary.histogram("c5_hr", c5_hr)
            tf.summary.histogram("s5_hr", s5_hr)
            tf.summary.histogram("uv", uv)
            tf.summary.histogram("a5", uv)
            tf.summary.histogram("uv2", uv)

    return img_warp3
