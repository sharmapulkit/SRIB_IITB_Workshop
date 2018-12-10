import numpy as np
import h5py

def multiply_matrices(m1, m2):
    if not (m1.shape == m2.shape):
        raise Exception('Matrices are not of the same type')
    prod = m1*m2
    return np.sum(prod)

#def save_weights():

def load_weights(weights_file):
    hf = h5py.File(weights_file, 'r')
    data = hf.get('dataset_name').value
