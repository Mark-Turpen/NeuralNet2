import math
import random as r
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoidPrime(x):
    return sigmoid(x)*(1-sigmoid(x))

def quadCost(net,start,m,X,Y):
    c=0
    for i in range(start,start+m):
        c=c+(mag(subV(Y[i],net.compute(X[i]))))**2
    c=c/(2*m)
    return c


class Network():
    def __init__(self,sizes):
        self.L=len(sizes)
        B=[]
        for i in range(0,self.L-1):
            B=B+[np.array(np.random.randn(sizes[i+1]))]
        self.b=B
        W=[]
        for i in range(0,self.L-1):
            W=W+[np.array(np.random.randn(sizes[i+1],sizes[i]))]
        self.w=W
    def compute(self,x):
        a=[]
        for i in range(0,len(x)):
            a=a+[x[i][0]]
        a=np.array(a)
        a=[a]
        for i in range(1,self.L):
            a=a+[sigmoid(np.dot(self.w[i-1],a[i-1])   +self.b[i-1])]
        return a[self.L-1]
    def update(self,start,m,X,Y,eta):
        dbAvg=[]
        for i in range(0,len(self.b)):
            dbAvgi=np.zeros([len(self.b[i])])
            dbAvg=dbAvg+[dbAvgi]
        dwAvg=[]
        for i in range(0,len(self.w)):
            dwAvgi=np.zeros([len(self.w[i]),len(self.w[i][0])])
            dwAvg=dwAvg+[dwAvgi]
        for l in range(start,start+m):
            a=[]
            z=[]
            y=[]
            for i in range(0,len(X[l])):
                a=a+[X[l][i][0]]
                z=z+[X[l][i][0]]
            for i in range(0,len(Y[l])):
                y=y+[Y[l][i][0]]
            y=np.array(y)
            a=np.array(a)
            z=np.array(z)
            y=[y]
            a=[a]
            z=[z]
            for i in range(1,self.L):
                a=a+[sigmoid(np.dot(self.w[i-1],a[i-1])   +self.b[i-1])]
                z=z+[np.dot(self.w[i-1],a[i-1])+self.b[i-1]]
            delta=(a[self.L-1]-y)*sigmoidPrime(z[self.L-1])
            delta=[delta[0]]
            for i in range(1,self.L-1):
                delta=delta+[np.dot(self.w[self.L-1-i].T,delta[i-1])*sigmoidPrime(z[self.L-1-i])]
            delta=delta[::-1]
            db=[]
            for i in range(0,len(self.b)):
                dbi=delta[i]
                db=db+[dbi]
            dw=[]
            for i in range(0,len(self.w)):
                dwi=np.outer(delta[i],a[i])
                dw=dw+[dwi]
            for i in range(0,len(self.b)):
                dbAvg[i]=dbAvg[i]+db[i]/m
            for i in range(0,len(self.w)):
                for j in range(0,len(self.w[i])):
                    for k in range(0,len(self.w[i][j])):
                        dwAvg[i][j][k]=dwAvg[i][j][k]+dw[i][j][k]/m
        for i in range(0,len(self.b)):
            self.b[i]=self.b[i]-eta*dbAvg[i]
        for i in range(0,len(self.w)):
            self.w[i]=self.w[i]-eta*dwAvg[i]




with np.load('mnist.npz') as data:
    X = data['training_images']
    Y = data['training_labels']
    XTest = data['test_images']
    YTest = data['test_labels']



n=50000
sizes=[784,30,10]
start=0
m=10
eta=3.0


net=Network(sizes)
t0=time.process_time()
for i in range(0,int(n/m)):
    net.update(start,m,X,Y,eta)
    start=start+m
    print("finished")
    print(i*m)
    print("out of")
    print(int(n))
print("\n")
print("it took")
print((time.process_time()-t0)/60)
print("minutes to train")



nTest=10000
right=0
t0=time.process_time()
for i in range(0,nTest):
    a1=net.compute(XTest[i])
    y1=YTest[i]
    y2=[]
    for j in range(0,len(a1)):
        y2=y2+[y1[j][0]]
    if (np.where(a1 ==max(a1))[0][0]==y2.index(max(y2))):
        right=right+1

print("\n")
print(right)
print(" / ")
print(nTest)
print("correct")
print("\n")
print("it took")
print((time.process_time()-t0)/60)
print("minutes to test")
print("\n")

pltN=5
import matplotlib as mpl
mpl.rcParams['text.color'] = 'red'
for i in range(0,pltN**2):
    p=plt.subplot(pltN,pltN,i+1)
    p.axes.get_xaxis().set_visible(False)
    p.axes.get_yaxis().set_visible(False)
    a1=net.compute(XTest[i])
    y1=YTest[i]
    y2=[]
    for j in range(0,len(a1)):
        y2=y2+[y1[j][0]]
    correct=y2.index(max(y2))
    guess=np.where(a1 ==max(a1))
    title="                   "+str(guess[0][0])+" "+str(correct)
    plt.title(title)
    plt.imshow(XTest[i].reshape(28,28),cmap='gray')

plt.show()
