import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    INITIALIZER = tf.truncated_normal_initializer(stddev=1e-2)
    REGULARIZER = tf.contrib.layers.l2_regularizer(1e-3)

    def _conv2d(inputs, filters, kernel_size, stride, name):
        return tf.layers.conv2d(inputs=inputs,
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=(stride, stride),
                                padding='same',
                                kernel_initializer=INITIALIZER,
                                kernel_regularizer=REGULARIZER,
                                name=name)

    def _conv2d_transpose(inputs, filters, kernel_size, stride, name):
        return tf.layers.conv2d_transpose(inputs=inputs,
                                          filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(stride, stride),
                                          padding='same',
                                          kernel_initializer=INITIALIZER,
                                          kernel_regularizer=REGULARIZER,
                                          name=name)

    # Decode Layer 7
    output = _conv2d(vgg_layer7_out, num_classes, 1, 1, 'layer7')
    output = _conv2d_transpose(output, num_classes, 4, 2, 'decoded_layer7')

    # Skip from Layer 4
    pool4_out = _conv2d(vgg_layer4_out, num_classes, 1, 1, 'pool4')
    pool4_out = tf.multiply(pool4_out, 1e-2, name='pool4_out_scaled')
    output = tf.add(output, pool4_out, name='skipped_pool4')
    output = _conv2d_transpose(output, num_classes, 4, 2, 'decoded_pool4')

    # Skip from Layer 3  
    pool3_out = _conv2d(vgg_layer3_out, num_classes, 1, 1, 'pool3')
    pool3_out = tf.multiply(pool3_out, 1e-4, name='pool3_out_scaled')
    output = tf.add(output, pool3_out, name='skipped_pool3')
    output = _conv2d_transpose(output, num_classes, 16, 8, 'decoded_pool3')

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    regularization_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)) + regularization_loss
    training_operation = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    KEEP_PROB = 0.8
    LEARNING_RATE = 1e-3

    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        print('EPOCH {}'.format(i+1), end='', flush=True)
        count = 0
        total_loss = 0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})
            total_loss = total_loss + loss
            count = count + 1
            print('.', end='', flush=True)
        print()
        print('Average Loss = {:.5f}'.format(total_loss/count))

tests.test_train_nn(train_nn)


def run():
    NUM_CLASSES = 2
    IMAGE_SHAPE = (160, 576)
    DATA_DIR = './data'
    RUNS_DIR = './runs'
    EPOCHS = 90
    BATCH_SIZE = 5

    tests.test_for_kitti_dataset(DATA_DIR)

    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as session:
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Build the network
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(session, vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
        logits, training_operation, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, NUM_CLASSES)

        # Train the network
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, training_operation, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, session, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
