"""

 2020 (c) piteren

"""
import tensorflow as tf


def qnn_model(
        name=               'qnn',
        input_size=         4,
        hidden_layers_size= [12],
        gamma=              0.9,
        iLR=                0.001,
        seed=               121,
        opt_class=          tf.train.GradientDescentOptimizer,
        **kwargs):

    with tf.variable_scope(name):

        q_target_PH = tf.placeholder(
            shape=  (None,input_size),
            dtype=  tf.float32)
        rew_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.float32)
        states_PH = tf.placeholder(
            shape=  (None,input_size),
            dtype=  tf.float32)
        enum_actions_PH = tf.placeholder(
            shape=  (None,2),
            dtype=  tf.int32)

        layer = states_PH
        for l in hidden_layers_size:
            layer = tf.layers.dense(
                inputs=             layer,
                units=              l,
                activation=         tf.nn.relu,
                kernel_initializer= tf.contrib.layers.xavier_initializer(seed=seed))
        output = tf.layers.dense(
            inputs=             layer,
            units=              input_size,
            activation=         None,
            kernel_initializer= tf.contrib.layers.xavier_initializer(seed=seed))
        predictions = tf.gather_nd(output, indices=enum_actions_PH)
        labels = rew_PH + gamma * tf.reduce_max(q_target_PH, axis=1)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=         labels,
            predictions=    predictions))

    return {
        'q_target_PH':      q_target_PH,
        'rew_PH':           rew_PH,
        'states_PH':        states_PH,
        'enum_actions_PH':  enum_actions_PH,
        'output':           output,
        'loss':             loss}