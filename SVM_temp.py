import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm

X = np.loadtxt(fname="testdata.csv",dtype='float',skiprows=1,usecols=(0,1),delimiter=',')
Y = np.loadtxt(fname="testdata.csv",dtype= 'float',skiprows=1,usecols=(2),delimiter=',')

#X = np.loadtxt(fname="data1.csv",dtype='float',skiprows=5,usecols=(4,7),delimiter=',')
#Y = np.loadtxt(fname="data1.csv",dtype= 'unicode',skiprows=5,usecols=(1),delimiter=',')
"""
for _ in range(len(Y)):
    if '雨' in Y[_]:
        Y[_] = 1
    else:
        Y[_] = -1
"""    
Y = Y.astype(float)
clf = svm.SVC(kernel='linear',C=0.2)
clf.fit(X, Y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx= np.linspace(0,20)
yy= a*xx - (clf.intercept_[0])/w[1]

b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])

b = clf.support_vectors_[-1]

yy_up = a * xx + (b[1] - a * b[0])

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],s=80, facecolors='none')
plt.scatter(X[:, 0], X[:, 1], c=Y)

plt.axis('tight')
plt.show()
print(clf.support_vectors_)
print(clf.support_vectors_[0],clf.support_vectors_[-1])