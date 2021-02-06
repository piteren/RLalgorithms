"""

 2020 (c) piteren

"""
import tensorflow as tf


def policy_model(
        name=           'policynn',
        state_size=     4,
        num_of_actions= 2,
        hidden_layers=  (20,),
        **kwargs):

    with tf.variable_scope(name):
        states_PH = tf.placeholder(
            shape=  (None, state_size),
            dtype=  tf.float32,
            name=   'input_states')
        acc_rew_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.float32,
            name=   'accumulated_rewards')
        actions_PH = tf.placeholder(
            shape=  None,
            dtype=  tf.int32,
            name=   'actions')

        layer = states_PH
        for i in range(len(hidden_layers)):
            layer = tf.layers.dense(
                inputs=             layer,
                units=              hidden_layers[i],
                activation=         tf.nn.relu,
                kernel_initializer= tf.contrib.layers.xavier_initializer(),
                name=               f'hidden_layer_{i+1}')
        logits = tf.layers.dense(
            inputs=             layer,
            units=              num_of_actions,
            activation=         None,#tf.nn.tanh,
            kernel_initializer= tf.contrib.layers.xavier_initializer(),
            name=               'logits')
        action_prob = tf.nn.softmax(logits)
        log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions_PH)
        loss = tf.reduce_mean(acc_rew_PH * log_policy)

    return {
        'states_PH':    states_PH,
        'acc_rew_PH':   acc_rew_PH,
        'actions_PH':   actions_PH,
        'action_prob':  action_prob,
        'loss':         loss}