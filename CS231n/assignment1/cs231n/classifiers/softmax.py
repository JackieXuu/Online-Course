import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    max_scores = np.max(scores)
    scores -= max_scores
    correct_class = scores[y[i]]
    loss += -correct_class + np.log(np.sum(np.exp(scores)))
    for j in range(num_classes):
        dW[:,j] += X[i]*(-1*(j==y[i]) + np.exp(scores[j])/(np.sum(np.exp(scores))))
  loss = loss / num_train + 0.5 * reg * np.sum(W*W)
  dW = dW / num_train + reg * W
  #############################################################################
  #                    END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """      
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_features = W.shape[0]

  scores = X.dot(W)
  scores = (scores.T - np.max(scores, axis = 1)).T
  correct_class = scores[np.arange(num_train), y]
  loss = -np.sum(correct_class) + np.sum(np.log(np.sum(np.exp(scores), axis = 1)))
  loss = loss / num_train + 0.5 * reg * np.sum(W*W)

  scores_exp = np.exp(scores)
  sum = scores_exp / (np.sum(scores_exp, axis = 1).reshape(-1, 1))
  sum[np.arange(num_train), y] -= 1
  dW = (X.T).dot(sum) / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

