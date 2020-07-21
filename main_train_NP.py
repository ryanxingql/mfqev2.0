import gc, os, glob, argparse, h5py
import numpy as np
import tflearn
import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import reduce # for calculating PSNR
from operator import mul # for calculating the num of parameters
import net_MFCNN


def transformer(batch, chan, flow, U , out_size, name='SpatialTransformer', **kwargs):

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

            x = tf.cast(_repeat2(tf.range(0, width), height * num_batch), 'float32') + x * WIDTH
            y = tf.cast(_repeat2(_repeat(tf.range(0, height), width), num_batch), 'float32') + y * HEIGHT

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


def load_stack(type_process, ite_stack):
    """Load stack npy.

    type_process: "tra" or "val".
    ite_stack: start from 0."""
    stack_name = "stack_" + type_process + "_pre_" + str(ite_stack) + ".hdf5"
    pre_list = h5py.File(os.path.join(dir_stack, stack_name), 'r')['stack_pre'][:]
    print("pre loaded.")

    stack_name = "stack_" + type_process + "_cmp_" + str(ite_stack) + ".hdf5"
    cmp_list = h5py.File(os.path.join(dir_stack, stack_name), 'r')['stack_cmp'][:]
    print("cmp loaded.")

    stack_name = "stack_" + type_process + "_sub_" + str(ite_stack) + ".hdf5"
    sub_list = h5py.File(os.path.join(dir_stack, stack_name), 'r')['stack_sub'][:]
    print("sub loaded.")

    stack_name = "stack_" + type_process + "_raw_" + str(ite_stack) + ".hdf5"
    raw_list = h5py.File(os.path.join(dir_stack, stack_name), 'r')['stack_raw'][:]
    print("raw loaded.")

    return pre_list, cmp_list, sub_list, raw_list


def cal_MSE(img1, img2):
    """Calculate MSE of two images.

    img: [0,1]."""
    MSE = tf.reduce_mean(tf.pow(tf.subtract(img1, img2), 2.0))
    return MSE


def cal_PSNR(img1, img2):
    """Calculate PSNR of two images.

    img: [0,1]."""
    MSE = cal_MSE(img1, img2)
    PSNR = 10.0 * tf.log(1.0 / MSE) / tf.log(10.0)
    return PSNR


def main_train():
    """Train and evaluate model.

    Output: model_QPxx, record_train_QPxx."""

    ### Defind a session
    sess = tf.Session(config = config)

    ### Set placeholder
    x1 = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])  # pre
    x2 = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])  # cmp
    x3 = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])  # sub
    x5 = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])  # raw

    is_training = tf.placeholder_with_default(False, shape=()) # for BN training/testing. default testing.

    PSNR_0 = cal_PSNR(x2, x5) # PSNR before enhancement (cmp and raw)

    ### Motion compensation
    x1to2 = warp_img(tf.shape(x2)[0], x2, x1, False)
    x3to2 = warp_img(tf.shape(x2)[0], x2, x3, True)

    ### Flow loss
    FlowLoss_1 = cal_MSE(x1to2, x2)
    FlowLoss_2 = cal_MSE(x3to2, x2)
    flow_loss = FlowLoss_1 + FlowLoss_2

    ### Enhance cmp frames
    x2_enhanced = net_MFCNN.network(x1to2, x2, x3to2, is_training)

    MSE = cal_MSE(x2_enhanced, x5)
    PSNR = cal_PSNR(x2_enhanced, x5) # PSNR after enhancement (enhanced and raw)
    delta_PSNR = PSNR - PSNR_0

    ### 2 kinds of loss for 2-step training
    OptimizeLoss_1 = flow_loss + ratio_small * MSE  # step1: the key is MC-subnet.
    OptimizeLoss_2 = ratio_small * flow_loss + MSE  # step2: the key is QE-subnet.

    ### Defind optimizer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        Training_step1 = tf.train.AdamOptimizer(lr_ori).minimize(OptimizeLoss_1)
        Training_step2 = tf.train.AdamOptimizer(lr_ori).minimize(OptimizeLoss_2)

    ### TensorBoard
    tf.summary.scalar('MSE loss of motion compensation', flow_loss)
    tf.summary.scalar('MSE loss of final quality enhancement', MSE)
    tf.summary.scalar('MSE loss for training step1 (mainly MC-subnet)', OptimizeLoss_1)
    tf.summary.scalar('MSE loss for training step2 (mainly QE-subnet)', OptimizeLoss_2)
    tf.summary.scalar('PSNR before enhancement', PSNR_0)
    tf.summary.scalar('PSNR after enhancement', PSNR)
    tf.summary.scalar('PSNR improvement', delta_PSNR)

    tf.summary.image('cmp', x2)
    tf.summary.image('x1to2', x1to2)
    tf.summary.image('x3to2', x3to2)
    tf.summary.image('enhanced', x2_enhanced)
    tf.summary.image('raw', x5)

    summary_writer = tf.summary.FileWriter(dir_model, sess.graph)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=None) # define a saver

    sess.run(tf.global_variables_initializer()) # initialize network variables

    ### Calculate and present the num of parameters
    num_params = 0
    for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("# num of parameters: %d #" % num_params)
    file_object.write("# num of parameters: %d #\n" % num_params)
    file_object.flush()

    ### Find all stacks then cal their number
    stack_name = os.path.join(dir_stack, "stack_tra_pre_*")
    num_TrainingStack = len(glob.glob(stack_name))
    stack_name = os.path.join(dir_stack, "stack_val_pre_*")
    num_ValidationStack = len(glob.glob(stack_name))

    print("##### Start running! #####")

    num_TrainingBatch_count = 0

    ### Step 1: converge MC-subnet; Step 2: converge QE-subnet
    for ite_step in [1,2]:

        if ite_step == 1:
            num_epoch = epoch_step1
        else:
            num_epoch = epoch_step2

        ### Epoch by Epoch
        for ite_epoch in range(num_epoch):

            ### Train stack by stack
            for ite_stack in range(num_TrainingStack):

                pre_list, cmp_list, sub_list, raw_list = [], [], [], []
                gc.collect()
                pre_list, cmp_list, sub_list, raw_list = load_stack("tra", ite_stack)
                gc.collect()
                num_batch = int(len(pre_list) / BATCH_SIZE)

                ### Batch by batch
                for ite_batch in range(num_batch):

                    print("\rstep %1d - epoch %2d/%2d - training stack %2d/%2d - batch %3d/%3d" % \
                        (ite_step, ite_epoch+1, num_epoch, ite_stack+1, num_TrainingStack, ite_batch+1, num_batch), end="")

                    start_index = ite_batch * BATCH_SIZE
                    next_start_index = (ite_batch + 1) * BATCH_SIZE

                    if ite_step == 1:
                        Training_step1.run(session=sess, feed_dict={
                                              x1: pre_list[start_index:next_start_index],
                                              x2: cmp_list[start_index:next_start_index],
                                              x3: sub_list[start_index:next_start_index],
                                              x5: raw_list[start_index:next_start_index],
                                              is_training: True}) # train
                    else:
                        Training_step2.run(session=sess, feed_dict={
                                              x1: pre_list[start_index:next_start_index],
                                              x2: cmp_list[start_index:next_start_index],
                                              x3: sub_list[start_index:next_start_index],
                                              x5: raw_list[start_index:next_start_index],
                                              is_training: True})

                    # Update TensorBoard and print result
                    num_TrainingBatch_count += 1
                    
                    if ((ite_batch + 1) == int(num_batch / 2)) or ((ite_batch + 1) == num_batch):

                        summary, delta_PSNR_batch, PSNR_0_batch, FlowLoss_batch, MSE_batch = sess.run([summary_op, delta_PSNR, PSNR_0, flow_loss, MSE], feed_dict={
                                        x1: pre_list[start_index:next_start_index],
                                        x2: cmp_list[start_index:next_start_index],
                                        x3: sub_list[start_index:next_start_index],
                                        x5: raw_list[start_index:next_start_index],
                                        is_training: False})

                        summary_writer.add_summary(summary, num_TrainingBatch_count)
                        
                        print("\rstep %1d - epoch %2d - imp PSNR: %.3f - ori PSNR: %.3f - MSE loss of MC: %.5f - MSE loss of QE: %.8f" % \
                            (ite_step, ite_epoch+1, delta_PSNR_batch, PSNR_0_batch, FlowLoss_batch, MSE_batch))
                        file_object.write("step %1d - epoch %2d - imp PSNR: %.3f - ori PSNR: %.3f - MSE loss of MC: %.5f - MSE loss of QE: %.8f\n" % \
                            (ite_step, ite_epoch+1, delta_PSNR_batch, PSNR_0_batch, FlowLoss_batch, MSE_batch))
                        file_object.flush()


            ### Store the model of this epoch
            if ite_step == 1:
                CheckPoint_path = os.path.join(dir_model, "model_step1.ckpt")
            else:
                CheckPoint_path = os.path.join(dir_model, "model_step2.ckpt")
            saver.save(sess, CheckPoint_path, global_step=ite_epoch)

            sum_improved_PSNR = 0
            num_patch_count = 0

            ### Eval stack by stack, and report together for this epoch
            for ite_stack in range(num_ValidationStack):

                pre_list, cmp_list, sub_list, raw_list = [], [], [], []
                gc.collect()
                pre_list, cmp_list, sub_list, raw_list = load_stack("val", ite_stack)
                gc.collect()

                num_batch = int(len(pre_list) / BATCH_SIZE)

                ### Batch by batch
                for ite_batch in range(num_batch):

                    print("step %1d - epoch %2d/%2d - validation stack %2d/%2d                " % \
                        (ite_step, ite_epoch+1, num_epoch, ite_stack+1, num_ValidationStack))

                    start_index = ite_batch * BATCH_SIZE
                    next_start_index = (ite_batch + 1) * BATCH_SIZE

                    delta_PSNR_batch = sess.run(delta_PSNR, feed_dict={
                                        x1: pre_list[start_index:next_start_index],
                                        x2: cmp_list[start_index:next_start_index],
                                        x3: sub_list[start_index:next_start_index],
                                        x5: raw_list[start_index:next_start_index],
                                        is_training: False})

                    sum_improved_PSNR += delta_PSNR_batch * BATCH_SIZE
                    num_patch_count += BATCH_SIZE

            if num_patch_count != 0:
                print("### imp PSNR by model after step %1d - epoch %2d/%2d: %.3f ###" % \
                    (ite_step, ite_epoch+1, num_epoch, sum_improved_PSNR/num_patch_count))
                file_object.write("### imp PSNR by model after step %1d - epoch %2d/%2d: %.3f ###\n" % \
                    (ite_step, ite_epoch+1, num_epoch, sum_improved_PSNR/num_patch_count))
                file_object.flush()


if __name__ == '__main__':

    ### Settings
    CHANNEL = 1 # use only Y

    ratio_small = 0.01
    lr_ori = 1e-4
    epoch_step1 = 20
    epoch_step2 = 60

    parser = argparse.ArgumentParser()
    parser.add_argument('-hf', '--height', type=int, help="HEIGHT of frame")
    parser.add_argument('-wf', '--width', type=int, help="WIDTH of frame")
    parser.add_argument('-bs', '--batch_size', type=int)
    parser.add_argument('-gpu', '--gpu', type=str, help="GPU")
    parser.add_argument('-qp', '--qp', type=str, help="QP")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    QP = args.qp
    WIDTH = args.width
    HEIGHT = args.height

    dir_stack = "/home/x/SCI_1/MFQEv2.0/Database/Training_stack/QP" + QP
    dir_model = "./model_QP" + QP
    record_FileName = "./record_train_QP" + QP + ".txt"
    file_object = open(record_FileName, 'w')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only show error and warning
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = tf.ConfigProto(allow_soft_placement = True)   # if GPU is not usable, then turn to CPU automatically

    main_train()

    print("##### Training completes! #####")
    file_object.write("##### Training completes! #####\n")

    file_object.close()

