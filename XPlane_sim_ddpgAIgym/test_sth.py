import tensorflow as tf
action = [0, -1.2, 2, 3, 1, 0]
action = tf.nn.sigmoid(action)
print(action)