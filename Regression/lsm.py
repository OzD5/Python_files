import numpy as np
import matplotlib.pyplot as plt

# Linear Least Squares Method
# This code finds the best fit line for a set of points using the least squares method via matrix operations on numpy.
points=[(1,2),(2,4),(3,9),(4,10),(10,20),(11,22),(20,49)]

x=[]
y=[]
for x1,y1 in points:
    x.append(x1)
    y.append(y1)
#X=[[1,x1],[1,x2]...[1,xn]] "1+1*X"
X=np.concatenate((np.transpose([np.ones(len(points))]),np.transpose([x])),axis=1)
#XTXINV=XT*X
XTX=(np.matmul(np.transpose(X),X))
#Inverting X^TX matrix
XTXinv=np.linalg.inv(XTX)
# ab=y*X*(XT*X)^-1
ab=np.matmul(np.matmul(y,X),XTXinv)

a=round(ab[1],4)
b=round(ab[0],4)


xaxis=np.linspace(-10,40)
yaxis=a*xaxis+b

plt.scatter(x,y,label = "Points")
plt.plot(xaxis,yaxis,label=f"y={a}x+{b}")
plt.legend()
plt.grid(True)
plt.show()

print(f"{ab[1]}x + {ab[0]}")