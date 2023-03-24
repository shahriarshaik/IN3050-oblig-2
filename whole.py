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



















from sklearn.model_selection import train_test_split




class NumpyLogReg(NumpyClassifier):

    def __init__(self, bias=-1, scaler=None):
        self.bias = bias
        self.scaler = scaler
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, t_train, eta=0.1, epochs=10, X_val=None, t_val=None, tol=1e-5, n_epochs_no_update=5):
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            if X_val is not None:
                X_val = add_bias(X_val, self.bias)

        (N, m) = X_train.shape
        self.weights = weights = np.zeros(m)
        no_update_count = 0
        last_loss = float("inf")

        for e in range(epochs):
            y = self.sigmoid(X_train @ weights)
            weights -= eta / N * X_train.T @ (y - t_train)

            # Calculate loss and accuracy at the end of each epoch
            loss = -np.mean(t_train * np.log(y) + (1 - t_train) * np.log(1 - y))
            self.losses.append(loss)
            acc = np.mean((y > 0.5) == t_train)
            self.accuracies.append(acc)

            # Calculate loss and accuracy for validation set (if provided)
            if X_val is not None and t_val is not None:
                val_loss, val_acc = self._validation_loss_accuracy(X_val, t_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

            # Check for early stopping
            if np.abs(last_loss - loss) < tol:
                no_update_count += 1
            else:
                no_update_count = 0

            if no_update_count >= n_epochs_no_update:
                break

            last_loss = loss

        self.total_epochs = e + 1



    def predict(self, X, threshold=0.5, apply_scaler=True, apply_bias=True):
        prob = self.predict_probability(X, apply_scaler=apply_scaler, apply_bias=apply_bias)
        return (prob > threshold).astype(int)



    def predict_probability(self, X, apply_scaler=True, apply_bias=True):
        if apply_scaler and self.scaler is not None:
            X = self.scaler(X)
    
        if self.bias and apply_bias:
            X = add_bias(X, self.bias)
    
        return self.sigmoid(X @ self.weights)
    
    def _validation_loss_accuracy(self, X_val, t_val):
        y_val = self.sigmoid(X_val @ self.weights)
        val_loss = -np.mean(t_val * np.log(y_val) + (1 - t_val) * np.log(1 - y_val))
        val_acc = np.mean((y_val > 0.5) == t_val)
        return val_loss, val_acc



# Assuming X and t are the feature matrix and target labels
X_train, X_val, t_train, t_val = train_test_split(X, t_multi, test_size=0.2, random_state=42)

etas = np.linspace(0.01, 1.4, num=200)
tols = [1e-4, 1e-5, 1e-6]
scalers = [standard_scaler, min_max_scaler]

best_acc = 0
best_eta = None
best_tol = None
best_scaler = None
best_clf = None

for eta in etas:
    for tol in tols:
        for scaler in scalers:
            X_train_scaled = scaler(X_train)
            X_val_scaled = scaler(X_val)
            cl = NumpyLogReg(scaler=scaler)
            cl.fit(X_train_scaled, t_train, eta=eta, epochs=200, X_val=X_val_scaled, t_val=t_val, tol=tol)
            
            acc = np.mean(cl.predict(X_val_scaled) == t_val)
            if acc > best_acc:
                best_acc = acc
                best_eta = eta
                best_tol = tol
                best_scaler = scaler.__name__
                best_clf = cl



print(f"Best hyperparameters found:\nScaler: {best_scaler}\neta: {best_eta}\ntol: {best_tol}\naccuracy: {best_acc}")


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

ax[0].plot(best_clf.losses, label='Training loss')
ax[0].plot(best_clf.val_losses, label='Validation loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation loss')
ax[0].legend()

ax[1].plot(best_clf.accuracies, label='Training accuracy')
ax[1].plot(best_clf.val_accuracies, label='Validation accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Training and Validation accuracy')
ax[1].legend()

plt.show()












class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 6):
        """Intialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden
        
        def logistic(x):
            return 1/(1+np.exp(-x))
        self.activ = logistic
        
        def logistic_diff(y):
            return y * (1 - y)
        self.activ_diff = logistic_diff
        
    def fit(self, X_train, t_train, eta=0.001, epochs = 100):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Itilaize the wights
        self.weights1 = (np.random.rand(
            dim_in + 1, 
            self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(
            self.dim_hidden+1, 
            dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        X_train_bias = add_bias(X_train, self.bias)
        
        for e in range(epochs):
            # One epoch
            hidden_outs, outputs = self.forward(X_train_bias)
            # The forward step
            out_deltas = (outputs - T_train)
            # The delta term on the output node
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The delta terms at the output of the jidden layer
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  
            # The deltas at the input to the hidden layer
            self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas 
            # Update the weights
            
    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = hidden_outs @ self.weights2
        return hidden_outs, outputs
    
    def predict(self, X):
        """Predict the class for the mebers of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)

