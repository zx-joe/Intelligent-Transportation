import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  loss = 0.0
  dW = np.zeros_like(W)

  N = X.shape[0]
  f = np.dot(X, W)
  f_max = f.max(axis = 1).reshape(N, 1)
  
  f = f - f_max
  
  loss = np.log(np.exp(f).sum(axis = 1)).sum() - f[range(N), y].sum()

  total = np.exp(f) / (np.exp(f).sum(axis = 1)).reshape(N, 1)
  total[range(N), y] = total[range(N), y]-1
  dW = np.dot(X.T, total)

  loss = loss / N 
  dW = dW / N 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

