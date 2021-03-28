import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
num_of_actions = env.action_space.n

print(state_size, num_of_actions)

end_game_reward = 100

# critic
class QNetwork:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions):
        self.actions = tf.placeholder(shape=(None, num_of_actions), dtype=tf.float32, name='dqn_actions')
        self.future_actions = tf.placeholder(shape=(None, num_of_actions), dtype=tf.float32, name='dqn_future_actions')
        self.q_target = tf.placeholder(shape=(None, num_of_actions), dtype=tf.float32, name='dqn_q_target')
        self.r = tf.placeholder(shape=None, dtype=tf.float32, name='dqn_r')
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='dqn_states')

        _layer = self.states
        for l in hidden_layers_size:
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.last_layer = tf.layers.dense(inputs=_layer, units=num_of_actions, activation=None,  # Linear activation
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.predictions = tf.reduce_sum(self.last_layer * self.actions, axis=1)
        self.labels = self.r + gamma * tf.reduce_sum(self.q_target * self.future_actions, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.labels, predictions=self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

# actor
class PolicyGradient:
    def __init__(self, hidden_layers_size, gamma, learning_rate, input_size, num_of_actions):
        self.states = tf.placeholder(shape=(None, input_size), dtype=tf.float32, name='pg_states')
        self.q = tf.placeholder(shape=None, dtype=tf.float32, name='pg_q')
        self.actions = tf.placeholder(shape=(None, num_of_actions), dtype=tf.float32, name='pg_actions')

        _layer = self.states
        for l in hidden_layers_size:
            _layer = tf.layers.dense(inputs=_layer, units=l, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.last_layer = tf.layers.dense(inputs=_layer, units=num_of_actions, activation=None,  # Linear activation
                                          kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.action_prob = tf.nn.softmax(self.last_layer)
        self.log_policy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.last_layer, labels=self.actions)
        self.cost = tf.reduce_mean(self.q * self.log_policy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)


class ActorCritic:
    def __init__(self, input_size, num_of_actions, actor_hidden_layers_size, critic_hidden_layers_size,
                 actor_gamma, critic_gamma, actor_learning_rate, critic_learning_rate):
        self.critic = QNetwork(critic_hidden_layers_size, critic_gamma, critic_learning_rate, input_size,
                               num_of_actions)
        self.actor = PolicyGradient(actor_hidden_layers_size, actor_gamma, actor_learning_rate, input_size,
                                    num_of_actions)
        self.memory = []
        self.num_of_actions = num_of_actions

    def _extract_from_batch(self, batch, key):
        return np.array(list(map(lambda x: x[key], batch)))

    def _one_hot_encoding(self, arr):
        hot = np.zeros((len(arr), self.num_of_actions))
        hot[np.arange(len(arr)), arr] = 1
        return hot

    def _update_terminal_states(self, q_target, terminals):
        for i in range(len(terminals)):
            if terminals[i]:
                q_target[i] = 0.0
        return q_target

    def remember(self, **kwargs):
        self.memory.append(kwargs)

    def actions_prob(self, session, state):
        return session.run(self.actor.action_prob, feed_dict={self.actor.states: np.expand_dims(state, axis=0)}).flatten()

    def act(self, session, state, greedy=False):
        actions_prob = self.actions_prob(session, state)
        if greedy:
            action = np.argmax(actions_prob)
        else:
            action = np.random.choice(self.num_of_actions, p=actions_prob)
        return action

    def learn(self, session):
        batch = shuffle(self.memory)
        next_states = self._extract_from_batch(batch, 'next_state')
        states = self._extract_from_batch(batch, 'state')
        rewards = self._extract_from_batch(batch, 'reward')
        actions = self._extract_from_batch(batch, 'action')
        terminals = self._extract_from_batch(batch, 'game_over')

        actions = self._one_hot_encoding(actions)
        q = session.run(self.critic.last_layer, feed_dict={self.critic.states: states})
        q_t = session.run(self.critic.last_layer, feed_dict={self.critic.states: next_states})
        q_t = self._update_terminal_states(q_t, terminals=terminals)

        next_actions = session.run(self.actor.action_prob, feed_dict={self.actor.states: next_states})

        actor_cost, _, critic_cost, _ = session.run([self.actor.cost, self.actor.optimizer,
                                                     self.critic.cost, self.critic.optimizer],
                                                    feed_dict={self.actor.states: states,
                                                               self.actor.q: q,
                                                               self.actor.actions: actions,
                                                               self.critic.q_target: q_t,
                                                               self.critic.r: rewards,
                                                               self.critic.actions: actions,
                                                               self.critic.future_actions: next_actions,
                                                               self.critic.states: states})
        if np.isnan(actor_cost) or np.isnan(critic_cost): raise Exception('NaN cost!')
        self.memory = []
        return actor_cost, critic_cost

actor_hidden_layers = [24]
critic_hidden_layers = [24,24]
learning_rate = 0.001
gamma = 0.99


sess = tf.Session()
ac = ActorCritic(input_size=state_size, num_of_actions=num_of_actions,
                 actor_hidden_layers_size=actor_hidden_layers, critic_hidden_layers_size=critic_hidden_layers,
                 actor_gamma=gamma, critic_gamma=gamma,
                 actor_learning_rate=learning_rate, critic_learning_rate=learning_rate)

game_df = pd.DataFrame(columns=['game','steps','actor_cost','critic_cost'])
sess.run(tf.global_variables_initializer())


def print_stuff(s, every=50):
    if game % every == 0 or game == 1:
        print(s)

games = 1000

for g in range(games):
    game = g + 1
    game_over = False
    next_state = env.reset()
    steps = 0
    while not game_over:
        steps += 1
        state = np.copy(next_state)
        action = ac.act(sess, state)
        next_state, r, game_over, _ = env.step(action)
        if game_over and steps < env._max_episode_steps: r = end_game_reward
        ac.remember(state=state, action=action, reward=r, next_state=next_state, game_over=game_over)
        actor_cost, critic_cost = ac.learn(sess)
    print_stuff('Game {g} ended after {s} steps | Actor cost: {a:.2e}, Critic cost: {c:.2e}'.format(g=game, s=steps, a=actor_cost, c=critic_cost))
    game_df = game_df.append({'game':game, 'steps':steps, 'actor_cost':actor_cost, 'critic_cost':critic_cost},
                             ignore_index=True)

game_df['steps_moving_average'] = game_df['steps'].rolling(window=50).mean()
ax = game_df.plot('game','steps_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Steps')
plt.show()

game_df['actor_cost_moving_average'] = game_df['actor_cost'].rolling(window=50).mean()
ax = game_df.plot('game','actor_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Actor Cost')
plt.show()

game_df['critic_cost_moving_average'] = game_df['critic_cost'].rolling(window=50).mean()
ax = game_df.plot('game','critic_cost_moving_average', figsize=(10,10), legend=False)
ax.set_xlabel('Game')
ax.set_ylabel('Critic Cost')
plt.show()


next_state = env.reset()
env.render()
game_over = False
steps = 0
while not game_over:
    steps += 1
    state = np.copy(next_state)
    action = ac.act(sess, state, greedy=True)
    action_prob = ac.actions_prob(sess, state).tolist()
    next_state, _, game_over, _ = env.step(action)
    env.render()
print('Ended after {} steps'.format(steps))