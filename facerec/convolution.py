from __future__ import print_function
from math import ceil
import numpy as np
import tensorflow as tf
import os
import time
from facerec import functions as fn


def weight_variable(dims, name):
    initial = tf.truncated_normal(dims, stddev=0.1)
    w = tf.Variable(initial, name=name)
    tf.add_to_collection('weights', w)
    return w


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial, name=name)
    tf.add_to_collection('biases', b)
    return b


def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[stride, stride, stride, stride], padding='SAME', name="conv")


def max_pool(x, dim):
    return tf.nn.max_pool(x, ksize=[1, dim, dim, 1], strides=[1, 2, 2, 1], padding='SAME', name="max_pool")


def conv_layer(x, filter_size, num_filters, num_channels):
    """
    Conv layer performs 2D-convolution on input image with filter size defined by @filter arg,
    must also tell the width of the 3rd dimension of the input (e.g. number of colour channels in iamge,
    or number of filters from conv layer).

    2x2 max pooling is done after conv operation.

    Args:
        x: input matrix
        filter_size: width of filter (px)
        num_filters: number of convolutional filters to apply (= new 3rd dimension size)
        num_channels: size of input 3rd dimension

    Returns:
        max-pooled conv layer
    """
    W_conv = weight_variable([filter_size, filter_size, num_channels, num_filters], 'weightConv')
    b_conv = bias_variable([num_filters], 'biasConv')

    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)  # First convolution, stride of 1, filter size 5
    h_pool = max_pool(h_conv, 2)

    return h_pool


def fc_layer(x, shape, activation=None):
    """
    Fully connected layer with optional activation

    Args:
        x: input vector
        shape: shape of layer (1st dim is input size, 2nd dim is output size)
        activation: function to apply to output, if None, nothing is applied, just simple h = Wx + b

    Returns:
        output from fully connected layer
    """
    W_fc = weight_variable(shape, 'weightFC')
    b_fc = bias_variable([shape[1]], 'biasFC')

    h_fc = tf.matmul(x, W_fc) + b_fc

    if activation is not None:
        h_fc = activation(h_fc)

    return h_fc


def conv_net(x, y, size=None, train=True, save_dir="~/cnn_data", epochs=25, batch_dim=256):
    with tf.Graph().as_default():
        num_channels = x.shape[3]
        num_classes = y.shape[1]

        save_dir = fn.handle_filepath(save_dir)

        width = 64
        height = 80

        if size is not None:
            height = size[0]
            width = size[1]
            num_channels = size[2] or 1

        x_in = tf.placeholder(tf.float32, [None, height, width, num_channels], name="x_in")
        y_in = tf.placeholder(tf.float32, [None, num_classes], name="y_in")
        keep_prob = tf.placeholder(tf.float32)

        batch_size = tf.shape(x_in)[0]

        conv1 = conv_layer(x_in, filter_size=5, num_filters=64, num_channels=num_channels)

        conv2 = conv_layer(conv1, filter_size=5, num_filters=64, num_channels=64)

        conv2_flat = tf.reshape(conv2, [batch_size, -1])
        shape = (height * width)/16 * 64  # Get number of units in flattened conv output.

        fc1 = fc_layer(conv2_flat, [shape, 1024], activation=tf.nn.relu)

        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        y_conv = fc_layer(fc1_drop, [1024, num_classes])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_conv, labels=y_in, name='cross_entropy_per_example'))
        tf.summary.scalar("cross_entropy", cross_entropy)

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        prediction_output = tf.argmax(tf.nn.softmax(y_conv), 1)  # Get vector of predicted labels.

        summaries = tf.summary.merge_all()  # Get the summaries of all values being tracked

        saver = tf.train.Saver({v.op.name: v for v in tf.global_variables()}, save_relative_paths=True)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(save_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            if train:
                length = len(x)
                p_count = 1
                step = ceil(length / batch_dim)
                steps = int(step * epochs)
                print_step = ceil(steps / 20.0)

                net_start = time.time()
                for i in range(steps):
                    start_index = int((i % step) * batch_dim)
                    end_index = start_index + batch_dim
                    ind = list(range(start_index, end_index+1))
                    image_batch = x.take(ind, axis=0, mode='wrap')
                    batch_labels = y.take(ind, axis=0, mode='wrap')

                    st = time.time()
                    _, loss, summary = sess.run([train_step, cross_entropy, summaries],
                                                feed_dict={x_in: image_batch, y_in: batch_labels, keep_prob: 0.5})
                    e = time.time()

                    diff = e - st
                    eta = int(diff * (steps - i))

                    if i % 10 == 0:
                        train_writer.add_summary(summary, i)

                    if i % print_step == 0:
                        print("\r%5d/%d [%s%s] - ETA: %3ds - loss: %.6f" %
                              (i, steps, "=" * (p_count - 1) + ">", "." * (20 - p_count), eta, loss), end='')
                        p_count += 1

                    if i % 50 == 0:
                        saver.save(sess, os.path.join(save_dir, "model.ckpt"))

                net_end = time.time()
                print("> Trained CNN in %.4fsecs" % (net_end - net_start))

                saver.save(sess, os.path.join(save_dir, "model.ckpt"))
                print ("Training CNN done")

            else:
                ckpt = tf.train.get_checkpoint_state(save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise IOError("No checkpoint found in file %s ..." % save_dir)

                # Now run the model to get predictions
                predictions = []
                for i in range(int(ceil(float(len(x)) / batch_dim))):
                    start_index = i * batch_dim
                    end_index = (i * batch_dim + batch_dim)
                    if end_index > len(x):
                        end_index = len(x)
                    x_batch = x[start_index : end_index]
                    y_batch = y[start_index : end_index]
                    pred = prediction_output.eval(feed_dict={x_in: x_batch, y_in: y_batch, keep_prob: 1.0})
                    predictions.append(pred)

                predictions = np.concatenate(predictions, axis=0)

                return predictions
