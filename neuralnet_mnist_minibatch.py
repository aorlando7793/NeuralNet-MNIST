import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Process data
df = pd.read_csv('train.csv')
data = df.as_matrix()
# normalize data
for i in xrange(data.shape[1] - 1):
    if data[:20000,i+1].std() != 0:
        data[:20000,i+1] = (data[:20000,i+1] - data[:20000,i+1].mean())/data[:20000,i+1].std()
# set test and training data
Xtrain = data[:10000, 1:]
Ttrain = data[:10000, 0]
Xtest = data[10000:20000, 1:]
Ttest = data[10000:20000, 0]

Ntrain, D = Xtrain.shape
Ntest = Xtest.shape[0]        
K = len(set(Ttrain))

Ttrain_1hot = np.zeros((Ntrain, K))
Ttrain_1hot[xrange(Ntrain), Ttrain] = 1 
Ttest_1hot = np.zeros((Ntest, K))
Ttest_1hot[xrange(Ntest), Ttest] = 1 


# Define functions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)
    
def forward(X, W1, b1, W2, b2):
    """Feed-forward function: takes in X, weights, and biases, returns P and Z
    """
    Z = np.tanh(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)
    return Y, Z

def predict(Y):
    """finds which class has highest probability for each sample
    """
    return np.argmax(Y, axis=1)
         
def cross_entropy(T, Y):
    return -np.mean(T*np.log(Y))
    
def classification_rate(T, Y):
    return np.mean(Y == T)
    

# set hyperparameters
M = 250
learning_rate = .3
epochs = 10
batch_size = 20

# randomly initialize weights
W1 = (D**(-1/2)) * np.random.randn(D, M)
b1 = np.zeros(M)
W2 = (M**(-1/2)) * np.random.randn(M, K)
b2 = np.zeros(K)
    
# pre-allocate cost lists
train_costs = []
test_costs = []

#time gd
ti = time.time()

# start mini batch gradient descent and backpropagation
print "Starting Gradient Descent"
for epoch in xrange(epochs):    
    mini_step = 0
    for i in xrange((Ntrain/batch_size) - 1):
        mb_Xtrain = Xtrain[mini_step:mini_step + batch_size, :]
        mb_Xtest = Xtest[mini_step:mini_step + batch_size, :]
        mb_Ytrain, mb_Ztrain = forward(mb_Xtrain, W1, b1, W2, b2)
        mb_Ytest, mb_Ztest = forward(mb_Xtest, W1, b1, W2, b2)
        
        mb_Ttrain_1hot = Ttrain_1hot[mini_step:mini_step + batch_size, :]
        mb_Ttest_1hot = Ttest_1hot[mini_step:mini_step + batch_size, :]
    
        ctrain = cross_entropy(mb_Ttrain_1hot, mb_Ytrain)
        ctest = cross_entropy(mb_Ttest_1hot, mb_Ytest)
        train_costs.append(ctrain)
        test_costs.append(ctest)
    
        W2 -= (learning_rate / float(batch_size)) * mb_Ztrain.T.dot(mb_Ytrain - mb_Ttrain_1hot)
        b2 -= (learning_rate / float(batch_size)) * (mb_Ytrain - mb_Ttrain_1hot).sum()
        dZ = (mb_Ytrain - mb_Ttrain_1hot).dot(W2.T) * (1 - mb_Ztrain*mb_Ztrain)
        W1 -= (learning_rate / float(batch_size)) * mb_Xtrain.T.dot(dZ)
        b1 -= (learning_rate / float(batch_size)) * dZ.sum(axis=0)
        
        mini_step += batch_size
    
    # shuffle data for next epoch
    np.random.shuffle(data[:10000, :])
    Xtrain = data[:10000, 1:]
    Ttrain = data[:10000, 0]
    Ttrain_1hot = np.zeros((Ntrain, K))
    Ttrain_1hot[xrange(Ntrain), Ttrain] = 1
    
    #if epoch % 50 == 0:
    #    print "epoch:", epoch

tf = time.time()
t = tf - ti

Ytrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
Ytest, Ztest = forward(Xtest, W1, b1, W2, b2)

print "Final train classification_rate:", classification_rate(Ttrain, predict(Ytrain))
print "Final test classification_rate:", classification_rate(Ttest, predict(Ytest))
print "Time Elapsed:", t/60, "Neurons:", M, "Learning Rate:", learning_rate
print "Epochs:", epochs, "Batch Size:", batch_size

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
    
        
        
    
    
    
    

