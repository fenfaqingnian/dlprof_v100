import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


def reformat(labels):
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return labels

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #print("shape of layer_1 matrix :" ,layer_1)
    layer_1 = tf.nn.relu(layer_1) 
          # activation function=rectifier , relu(features, name=none) same shape of the layer_1 matrix
    #print("shape after activation function of layer 1: ", layer_1)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    print("shape of out_layer matrix out of hidden layer --> out_layer :" , out_layer)
    return out_layer
# Accuracy
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
      / predictions.shape[0])
if __name__ == '__main__':
    # Importing the dataset
    dataset = pd.read_csv('Churn_Modelling.csv')
    print(dataset.head())
    # find out how many records we have
    len(dataset)
    #find out how many columns we have
    len(dataset.columns)
    X = dataset.iloc[:, 3:13].values # python index does not include the last column=13
    y = dataset.iloc[:, 13].values # but do it this way, it will only take column 13
    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling - in general, we need to always do feature scaling for neural network
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    records, input_num =X_train.shape

    num_labels = 2
    y_train = reformat(y_train)
    y_test = reformat(y_test)

    print('Training set', X_train.shape, y_train.shape)
    print('testing set', X_test.shape, y_test.shape)
    X_test.astype(np.float32)

    # improve your nn by inserting one hidden layer
    num_steps = 1000 # hyper-parameter
    batch_size = 100 # another hyper-parameter
    train_subset = 8000 # in case your memory capacity is low

    num_labels=2

    # neural network with 1 hidden layer 
    n_hidden_1 = 500 #  1st layer number of features, yet another hyper-parameter
    n_input = 10    #  data input (img shape: 28*28) since your input matrix shape is (train_subsetX28*28)=10000*784
    n_classes = 2    #  total classes 2:churn/not churn 


    # Store layers weight & bias using python dictionary-like syntax
    weights = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_classes]))
    } # you can access hidden layer1=h1's matrix via weights['h1'] as well as output from activation function via weight['out']

    biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1])),
        'out': tf.Variable(tf.zeros([n_classes]))
    }
    #print("shape of weight matrix h1: ", weights['h1'])
    #print("shape of weight matrix out : ", weights['out'])
    #print("shape of bias matrix b1 :", biases['b1'])
    #print("shape of bias matrix out", biases['out'])
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input]) # None means you can input dynamic amount of inputs images
    y = tf.placeholder(tf.float32, [None, n_classes]) 
    #print("x is of shape :", x)
    #print("y is of shape :", y)


    tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, 10))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.placeholder(tf.float32,
                                        shape=(None, 10))


    # Construct model 
    pred = multilayer_perceptron(x, weights, biases)
    valid_pred = multilayer_perceptron(tf_valid_dataset, weights, biases)


    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("Initialized tensorflow session")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = X_train[offset:(offset + batch_size), :]
            batch_labels = y_train[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {x : batch_data, y : batch_labels}
            _, l, predictions = sess.run([optimizer, loss, pred], feed_dict=feed_dict)
            if (step % 100 == 0):
                test_dict={x:X_test, y: y_test}
                _, l, testpred = sess.run([optimizer, loss, pred], feed_dict=test_dict)
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("test data accuracy: %.1f%%" % accuracy(testpred, y_test))