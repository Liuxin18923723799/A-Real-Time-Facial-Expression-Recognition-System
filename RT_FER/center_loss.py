import tensorflow as tf

slim = tf.contrib.slim

#tf.app.flags.DEFINE_boolean('center_loss', False, 'Use center loss.')

#FLAGS = tf.app.flags.FLAGS

def get_center_loss(features, labels, alpha, num_classes, name_scope):
    """get center loss

    Arguments:
        features: feature
        labels: the label for the feature
        alpha: control the learning rate of the center, should be a value in [0, 1]
        num_classes: the number of the class

    Return:
        loss: return the center loss
        centers: save the center of the sample
        centers_update_op: return the update_op, which is used to update the center position during the training process
    """
    len_features = features.get_shape()[1]
    centers = tf.get_variable(name_scope, [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])

    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(features - centers_batch)

    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op

'''
if FLAGS.center_loss is True:
        c_loss, centers, centers_update_op = center_loss.get_center_loss(end_points['features/teamid'], tf.argmax(labels, 1), 0.001, 3, name_scope='teamid_center')
        c_loss = 0.001 * c_loss
        tf.add_to_collection(tf.GraphKeys.LOSSES, c_loss)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, centers_update_op)
'''