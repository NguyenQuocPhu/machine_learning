import numpy as np
import matplotlib.pyplot as plt
# input data
Y = np.array([[49, 50, 51, 52, 54, 56, 58, 59, 60, 72, 63, 64, 66, 67, 68]]).T
X = np.array([[147, 150, 153, 155, 158, 160, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# build Xbar
# Xbar resemble [x1,x2,...,1]
ones = np.ones((X.shape[0],1))
Xbar = np.concatenate((X, ones),axis = 1 )
# draw axis and point
plt.plot(X,Y, 'ro' )
plt.axis([140, 185, 45, 70])
plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')


# 1/2 * || y - _X*w ||^2
# Derivative ("Dao ham" in Vietnamese)
# It's will become _X.T*(_X*w-y)
# To minimize, we will find Extreme ("Cuc tri" in Vietnamese) so the function will be _X.T*(_X*w-y)=0
# _X.T*_X*w = _X.T*y
# Call A = _X.T*_X, we will transfer it to "pseudo inverse" ("Gia nghich dao")(Remarkable: in python is numpy.linalg.pinv()), write by +A
# w = _X.T*y* + A
A = np.dot(Xbar.T, Xbar)
B = np.dot(Xbar.T, Y)

RR = np.linalg.pinv(A)
w = np.dot(np.linalg.pinv(A),B)

w_0 = w[0]
w_1 = w[1]
print(w)
# Draw the line 
x = np.arange(145, 185, 2)
y = w_0*x + w_1
plt.plot(x,y)
plt.show()
