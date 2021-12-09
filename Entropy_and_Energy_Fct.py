import numpy as np
import scipy.stats

#Define a functions, which calculates the normalized energy:
def normalized_energy_2d(signal):
  energy = np.sum(signal)
  energy = energy/(signal.shape[0]*signal.shape[1])
  return energy

#Define a function, which calculates the shannon entropy:
def entropy_shannon(signal, base = 2):
    _, counts = np.unique(signal, return_counts=True)
    return scipy.stats.entropy(counts, base = base)
