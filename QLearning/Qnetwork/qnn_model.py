"""

 2020 (c) piteren

"""
#import tensorflow as tf

from ptools.neuralmess.base_elements import tf
from ptools.neuralmess.layers import lay_dense

def qnn_model(
        name=               'qnn',
        num_actions=        4,
        num_states=         16,
        state_emb_width=    4,
        hidden_layers_size= (12,),
        gamma=              0.9,
        seed=               121,
        opt_class=          tf.train.GradientDescentOptimizer,
        **kwargs):

    with tf.variable_scope(name):

        q_target_PH = tf.placeholder(
            shape=  [None,num_actions],
            dtype=  tf.float32)
        rew_PH = tf.placeholder(
            shape=  [None],
            dtype=  tf.float32)
        state_PH = tf.placeholder(
            shape=  [None],
            dtype=  tf.int32)
        enum_actions_PH = tf.placeholder(
            shape=  [None,2],
            dtype=  tf.int32)

        state_emb = tf.get_variable(
            name=   'state_emb',
            shape=  [num_states,state_emb_width],
            dtype=  tf.float32)

        input = tf.nn.embedding_lookup(state_emb, state_PH)
        print('input:', input)

        for l in hidden_layers_size:
            input = lay_dense(
                input=      input,
                units=      l,
                activation= tf.nn.relu,
                seed=       seed)
        output = lay_dense(
            input=      input,
            units=      num_actions,
            activation= None,
            seed=       seed)
        predictions = tf.gather_nd(output, indices=enum_actions_PH)
        labels = rew_PH + gamma * tf.reduce_max(q_target_PH, axis=1)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=         labels,
            predictions=    predictions))

    return {
        'q_target_PH':      q_target_PH,
        'rew_PH':           rew_PH,
        'state_PH':         state_PH,
        'enum_actions_PH':  enum_actions_PH,
        'output':           output,
        'loss':             loss}