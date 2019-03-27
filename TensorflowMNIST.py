from tensorflow.examples.tutorials.mnist import input_data #gets mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #reads mnist data gets it ready
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784]) #PLACEHOLDER, changable variable, x will be changed later to input new data
W = tf.Variable(tf.zeros([784, 10])) #weight to individual pixels for prob of 0-9
b = tf.Variable(tf.zeros([10])) #adative weight to x*w +b
y = tf.nn.softmax(tf.matmul(x, W) + b) #softmax spits out probabilities between 0-1 for points in dataset, matmul multiplies x and W
y_ = tf.placeholder(tf.float32, [None, 10]) #will later be set as real values for test set to test and improve weights
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  #gets the loss of it ...somehow
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #tells graph best way to  improve, sets max change rate at .5, returns train_step which is the gradient descent operation
sess = tf.InteractiveSession() #starts interative tensorflow session
tf.global_variables_initializer().run() #initiallizes tensorflow variables ...and i think placeholders?
for _ in range(10000): #training step, trains 1000 times
  batch_xs, batch_ys = mnist.train.next_batch(100) # sets the batches of x and y equal to 100 random mnist samples. tests with samples of 1000
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #runs the session, uses train_step(which is an operation) to improve using batch_xs and batch_ys
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #argmax gives best tensor in an axis mybe?  gives back array of which guesses are right and which are wrong, ex[1,0,1,1], index 0 guess was right, 1 was wrong, 2 was right, 3 was right
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #finds percent wrong
#print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #gets the accuracy of the model

#test stuff
newx, batch_ys = mnist.train.next_batch(1)
feed_dict = {x: newx} #replace newx with array of 1 by 784 of an image
classification = sess.run(y, feed_dict)
#print (classification)

from PIL import Image
import os
import math
import numpy as np
openThis = input("which image file?\n")
if(openThis[len(openThis)-4] != "." and openThis[len(openThis)-5] != "."):
	openThis = openThis+".jpg"
im = Image.open(openThis)
pix = im.load()
imageSize = im.size
imageData = np.array([[0.0000 for x in range(784)]])
for length in range(imageSize[0]):
	for height in range(imageSize[1]):
		imageData[0][length+height*28] = abs(pix[length, height][0]-255)/255
#print(imageData)
newx, batch_ys = mnist.train.next_batch(1)
feed_dict = {x: imageData} #replace newx with array of 1 by 784 of an image
classification = sess.run(y, feed_dict)
print (classification)
for x in range(10):
	if(classification[0][x]>.5):
		print("is it... "+str(x))
#remember -
#linear model, links probability of individual pixel color value to probability of number 1-10
#  ex. middle pixel never in a 0, usually in a 1, sometimes in a 2