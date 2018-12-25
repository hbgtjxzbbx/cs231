import numpy as np
from random import shuffle
from past.builtins import xrange


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
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1),(num_train,1))
    f = f- f_max
    exp_f = np.exp(f)
    probs = exp_f/np.sum(exp_f, axis=1, keepdims=True)
    y_trueClass =np.zeros_like(probs)
    y_trueClass[np.arange(num_train),y] = 1.0


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    for i in range(num_train):
      for j in range(num_class):
        loss += -y_trueClass[i,j] * np.log(probs[i,j])
        dW[:,j] += (probs[i,j]-y_trueClass[i,j])*X[i,:]
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
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
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
    f = f - f_max
    exp_f = np.exp(f)
    probs = exp_f / np.sum(exp_f, axis=1, keepdims=True)
    y_trueClass = np.zeros_like(probs)
    y_trueClass[np.arange(num_train), y] = 1.0

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    loss = - np.sum(y_trueClass * np.log(probs))
    dW = (X.T).dot(probs-y_trueClass)
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W*W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
