def fullyconnected(ip, weights, bias):
  out = None
  C, W, H = ip.shape
  reshaped_input = ip.reshape((1, int(C*W*H)))
  out = np.dot(reshaped_input, weights) + bias
  return out