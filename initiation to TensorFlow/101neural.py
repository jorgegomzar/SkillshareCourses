# Learned from https://www.skillshare.com/classes/Neural-networks-for-beginners-from-scatch-in-tensorflow/2061762397
# All of this code is from that course

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Para que no muestre basura

W = tf.Variable([0.5], tf.float32) # Valor variable para entrenar
b = tf.Variable([-0.1], tf.float32) # Valor variable para entrenar
x = tf.placeholder(tf.float32) # Valor input

model = W * x + b # Model es la prediccion teorica
y = tf.placeholder(tf.float32) # Es el valor real, lo que debería dar

sqr_diff = tf.square(model - y) # Calcula la distancia entre el modelo teórico y el práctico
loss = tf.reduce_sum(sqr_diff) # Consigue un valor de pérdida total
optimizer = tf.train.GradientDescentOptimizer(0.01) # Utiliza el método de optimización por gradiente
training = optimizer.minimize(loss) # El entrenamiento se centra en minimizar la pérdida

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(loss, {x: [1,2,3], y: [5,6,7]})
    print(output)
    for i in range(1000):
        output = sess.run(loss, {x: [1,2,3], y: [5,6,7]})
        sess.run(training, {x: [1,2,3], y: [5,6,7]})
        print(sess.run([W, b]))

    print('TESTING')
    output = sess.run(loss, {x: [4,5,6], y: [8,9,10]})
    print(output)
