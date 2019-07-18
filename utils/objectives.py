import keras.metrics
import tensorflow as tf


def weighted_crossentropy(y_true, y_pred):

    class_weights = tf.constant([[[[1., 1., 10.]]]])  # seems has a useless 0 channel?

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    # if it is a boundary pixel, this weight will be large

    weighted_losses = weights * unweighted_losses

    loss = tf.reduce_mean(weighted_losses)

    return loss

