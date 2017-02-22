import numpy as np
import random
import os

class Buffer:
	"""
	The replay buffer used to decorolate batches of expert
	policy percepts.
	Args:
		name -- The name of the replay buffer used for saving.
		load -- Whether or not the replay buffer should reload an existing store.
	"""
	def __init__(self, name):
		self.name = name
		self.store = []

		self.save_dir = './replay/'
		if not os.path.exists(self.save_dir):
			os.mkdir(self.save_dir)

	def put(self, percept):
		"""
		Wraps the storage mechanism for the replay buffer.
		"""
		self.store += [percept]

	def size(self):
		return len(self.store)

	def get(self, num_samples):
		"""
		Gets NUM_SAMPLES samples from the replay buffer at random.
		"""
		return random.sample(self.store, num_samples)

	def load(self):
		"""
		Loads the replay buffer using self.NAME.
		"""
		self.store = (np.load(self.save_dir + str(self.name) + '.npz')['arr_0']).tolist()


	def save(self):
		"""
		Saves the replay buffer in a compressed format.
		"""
		np.savez(self.save_dir + str(self.name) + '.npz', np.array(self.store))