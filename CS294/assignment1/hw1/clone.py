import tensorflow as tf 
import numpy as np 
import tf_util
import os
import prettytensor as pt
import Buffer

REPLAY_START_SIZE=64
BATCH_SIZE=64
LEARNING_RATE=1e-3

class behavioural_clone:
	"""
	Behavioural cloning implementation
	"""

	def __init__(self, filename, state_shape, action_shape, sess, load=False):
		self.filename = filename
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.sess = sess
		self.save_dir = './bclone/{}'.format(filename)

		if not os.path.exists(self.save_dir):
			os.mkdir(self.save_dir)

		self.memory = Buffer.Buffer(filename)

		with tf.variable_scope("{}_clone".format(filename)) as clone_scope:
			self.state_tensor, self.action_tensor, self.expert_tensor, self.train_op = self.inference()

			scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=clone_scope.name)
			self.saver = tf.train.Saver(scope_vars)
			if load:
				self.load()
			else:
				init = tf.variables_initializer(scope_vars)
				self.sess.run(init)


	def inference(self):
		"""
		Construct the feed forward graph.
		"""		
		with tf.variable_scope("graph"):
			state_tensor = tf.placeholder(tf.float32, [None] + self.state_shape,name="state")
			action_tensor = (pt.wrap(state_tensor).reshape([-1] + [sum(self.state_shape)])
							.fully_connected(300, activation_fn=tf.nn.relu, l2loss=0.00001)
							.fully_connected(400, activation_fn=tf.nn.relu, l2loss=0.00001)
							.fully_connected(sum(self.action_shape), activation_fn=tf.identity))

		with tf.variable_scope("trainer"):
			expert_tensor = tf.placeholder(tf.float32, [None] + self.action_shape,name="expert")
			self.loss = 0.5*tf.reduce_mean(tf.square(expert_tensor - action_tensor))
			train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

		return state_tensor, action_tensor, expert_tensor, train_op


	def perceive(self, state, expert_action):
		"""
		Perceives a state and corresponding expert action pair.
		"""		
		self.memory.put((state, expert_action))
		return self.train(1)

	def train(self, iter):
		"""
		Trains the clone for an number of iterations on its perceived
		expert actions.
		"""		
		if self.memory.size() > REPLAY_START_SIZE:
			batch = self.memory.get(BATCH_SIZE)
			states, expert_actions = zip(*batch)

			_, loss = self.sess.run([self.train_op, self.loss], {
					self.state_tensor: np.array(states),
					self.expert_tensor: np.array(expert_actions)
				})
			return loss

	def action(self, state):
		"""
		Choose an action given a particular state.
		"""		
		batched_action = self.sess.run(self.action_tensor, {
				self.state_tensor: [state]
			})
		return batched_action[0]

	def save(self):
		self.saver.save(self.sess, self.save_dir+ "model.ckpt")
		self.memory.save()

	def load(self):
		self.saver.restore(self.sess, self.save_dir+ "model.ckpt")
		self.memory.load()