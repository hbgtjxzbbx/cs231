import numpy as np
import matplotlib.pyplot as plt

D = np.random.randn(1000, 500)
hidden_layer_sizes = [500] * 10
non_linearities = ['relu'] * len(hidden_layer_sizes)
act = {'relu': lambda x: np.maximum(0, x), 'tanh': lambda x: np.tanh(x)}
Hs = {}

for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i - 1]
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in, fan_out) /np.sqrt(fan_in/2)
    H = np.dot(X, W)
    H = act[non_linearities[i]](H)
    Hs[i] = H

# look at distributions at each layer

print('input layer had mean {} and std {}'.format(np.mean(D), np.std(D)))

layer_means = [np.mean(H[key]) for key in Hs]
layer_stds = [np.std(H[key]) for key in Hs]
for i in range(len(Hs)):
    print('hidden layer {} had mean {} and std {}'.format(i + 1, layer_means[i], layer_stds[i]))

# plot the means and standard deviations
plt.figure()
plt.subplot(121)
plt.plot(list(Hs.keys()), layer_means, 'ob-')
plt.title('layer mean')

plt.subplot(122)
plt.plot(list(Hs.keys()), layer_stds, 'or-')
plt.title('layer std')
plt.show()
# plot the raw distributions
plt.figure()
for i, H in enumerate(Hs):
    plt.subplot(1, len(Hs), i + 1)
    plt.hist(Hs[i].ravel(), 30, range=(-1, 1))
plt.show()
