import numpy as np
import matplotlib.pyplot as plt
import sklearn #for datasets

from sklearn.datasets import make_blobs
# Generate some blobs
X, t_multi = make_blobs(n_samples=[400,400,400, 400, 400], 
                        centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                        cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5],
                        n_features=2, random_state=2022)
# X: contains the coordinates of each data point in the dataset
# t_multi: contains the class label of each data point in the dataset

indices = np.arange(X.shape[0]) # create an array of indices for the rows of X
rng = np.random.RandomState(2022) # generate a random number generator
rng.shuffle(indices) # shuffle the indices
indices[:10] # print the first 10 indices


# Split the data into training, validation, and test sets
# Use the first 1000 examples for training
# Make a copy of the original data
X_train = X[indices[:1000],:]
# Extract the first 1000 rows of the target data
t_multi_train = t_multi[indices[:1000]]

# Use the next 500 examples for validation
X_val = X[indices[1000:1500],:]
t_multi_val = t_multi[indices[1000:1500]]

# Use the last 500 examples for testing
X_test = X[indices[1500:],:]
t_multi_test = t_multi[indices[1500:]]


# Convert each of the training, validation and test target vectors from the
# multi-class labels to binary labels representing the classes 3, 4 and 5.
t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')


plt.figure(figsize=(8,6)) # You may adjust the size
plt.scatter(X_train[:, 0], X_train[:, 1], c=t_multi_train, s=10.0)
plt.title('Multi-class set')


plt.figure(figsize=(8,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=t2_train, s=10.0)
plt.title('Binary set')

def add_bias(X, bias):
    """X is a Nxm matrix: N datapoints, m features
    bias is a bias term, -1 or 1. Use 0 for no bias
    Return a Nx(m+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 


class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""



class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias=bias
    
    def fit(self, X_train, t_train, eta = 0.1, epochs=10):
        """X_train is a Nxm matrix, N data points, m features
        t_train is avector of length N,
        the targets values for the training data"""
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        
        for e in range(epochs):
            weights -= eta / N *  X_train.T @ (X_train @ weights - t_train)      
    
    def predict(self, X, threshold=0.5):
        """X is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if self.bias:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold
    

def accuracy(predicted, gold):
    return np.mean(predicted == gold)


cl = NumpyLinRegClass()
cl.fit(X_train, t2_train)
accuracy(cl.predict(X_val), t2_val)


def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a make of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")

#    plt.show()


plot_decision_regions(X_train, t2_train, cl)




etas = [0.01, 0.05, 0.1, 0.5, 1.0]
#etas = np.linspace(0.01, 10.0, num=1000)
etas = np.linspace(0.01, 1.4, num=200) #best hittil

#epochs = [10, 20, 30, 50, 100, 200, 300, 500, 1000]
epochs = [10, 20, 30, 50, 100, 200]
epochs = np.linspace(0, 200, num=10, dtype=int)
print(epochs)

results = []

for eta in etas:
    for epoch in epochs:
        cl = NumpyLinRegClass()
        cl.fit(X_train, t2_train, eta=eta, epochs=epoch)
        acc = accuracy(cl.predict(X_val), t2_val)
        results.append((acc, eta, epoch))
        
results.sort(reverse=True)
best_acc, best_eta, best_epoch = results[0]

#print the top 5 results
print("Top 5 results:")
for acc, eta, epoch in results[:5]:
    print("eta={}, epochs={}, accuracy={}".format(eta, epoch, acc))
    

print("Accuracy for each set of hyperparameters:")
for acc, eta, epoch in results:
    #print("eta={}, epochs={}, accuracy={}".format(eta, epoch, acc))
    break
    
print("\nBest hyperparameters:")
print("eta={}, epochs={}, accuracy={}".format(best_eta, best_epoch, best_acc))

cl = NumpyLinRegClass()
cl.fit(X_train, t2_train, eta=best_eta, epochs=best_epoch)
plot_decision_regions(X_train, t2_train, cl)






class NumpyLinRegClass(NumpyClassifier):

    def __init__(self, bias=-1):
        self.bias = bias
    
    def fit(self, X_train, t_train, eta=0.1, epochs=10):
        """X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N,
        the target values for the training data"""
        
        if self.bias:
            X_train = add_bias(X_train, self.bias)
        
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        self.losses = []  # to store losses at each epoch
        self.accuracies = []  # to store accuracies at each epoch
        
        for e in range(epochs):
            ys = X_train @ weights
            loss = ((ys - t_train)**2).mean()  # calculate mean squared error loss
            accuracy1 = accuracy((ys > 0.5).astype(int), t_train)  # calculate accuracy
            
            weights -= eta / N *  X_train.T @ (ys - t_train)  # update weights
            
            self.losses.append(loss)
            self.accuracies.append(accuracy1)
            
            print(f"Epoch {e+1}: Loss={loss:.4f}, Accuracy={accuracy1:.4f}")

import matplotlib.pyplot as plt

# Normalize the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_normalized = (X_train - mean) / std


# Train the classifier with the best hyperparameters
cl = NumpyLinRegClass()
cl.fit(X_train_normalized, t2_train, eta=best_eta, epochs=best_epoch)

# Plot the loss and accuracy as a function of the number of epochs
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

ax[0].plot(cl.losses)
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training loss')

ax[1].plot(cl.accuracies)
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training accuracy')

plt.show()
