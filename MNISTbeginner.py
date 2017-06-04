import tensorflow as tf
import Tools
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
result = tf.argmax(y, 1)
goal = tf.argmax(y_, 1)

#random select 10 images
# test_xt, test_yt = mnist.train.next_batch(10)
test_xt = []
for j in range(1):
    # tmp = Tools.loadimage(str(j) + '.png')
    tmp = Tools.loadimage('testImage.png')
    test_xt.append(tmp)
    # Tools.saveimage(test_xt[j], str(j))

    # show image
    img = np.reshape(test_xt[j], [28, 28])
    cv2.imshow(str(j), img)
    cv2.moveWindow(str(j), 30 * j, 0)

#show result
print sess.run(result, feed_dict={x: test_xt})
# print sess.run(goal, feed_dict={y_: test_yt})

cv2.waitKey(0)
cv2.destroyAllWindows()