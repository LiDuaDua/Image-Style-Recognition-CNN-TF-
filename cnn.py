import tensorflow as tf
import csv
import numpy as np
from sklearn.model_selection import train_test_split

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def load_data(csvname):
    # load in data
    reader = csv.reader(open(csvname, "rb"), delimiter=",")
    d = list(reader)

    # import data and reshape appropriately
    data = np.array(d).astype("float")

    print 'shape+++++++++++++',data.shape
    train = data[:,0:785]

    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:784], data[:, 784:785], test_size = 0.33, random_state=42)

    print X_train.shape
    print y_train.shape

    train = np.concatenate((X_train, y_train), axis = 1)
    test = np.concatenate((X_test, y_test), axis = 1)

    return train, test

def next_batch(num, data):
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data[i] for i in idx]  # get list of `num` random samples
    data_shuffle = np.asarray(data_shuffle)  # get back numpy array

    return data_shuffle

train, test = load_data('labeledData.csv')
print("Download Done!")

sess = tf.InteractiveSession()

# paras
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([7 * 7 * 64, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([256, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 2])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

batchSize = 10

for i in range(100000):
    batch = next_batch(batchSize, train)
    batchY = np.zeros((batchSize, 2))
    trainY = np.zeros((len(train), 2))

    k = 0
    for j in train[:, 784]:
        trainY[k, int(j)] = 1
        k+=1

    k = 0
    for j in batch[:, 784]:
        batchY[k, int(j)] = 1
        k += 1

    if i % 100 == 0:
        print sess.run(b_fc2)
        train_accuacy = accuracy.eval(feed_dict={x: batch[:, 0:784], y_: batchY, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuacy))
    train_step.run(feed_dict = {x: batch[:, 0:784], y_: batchY, keep_prob: 0.5})

testY = np.zeros((len(test), 2))
k = 0
for j in test[:, 784]:
    testY[k, int(j)] = 1
    k+=1
# accuacy on test
print("test accuracy %g"%(accuracy.eval(feed_dict={x: test[:, 0:784], y_: testY, keep_prob: 1.0})))