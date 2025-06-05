import numpy as np
import matplotlib.pyplot as plt

# This code finds the best fit quadratic curve for a set of points using the least squares method via matrix operations on numpy.
# The user can click on the plot to add points, and then press 'd' to draw the best fit quadratic curve.
fig, ax = plt.subplots()
ax.set_title('Lisää pisteitä')
ax.set_xlim([-20, 20])
ax.set_ylim([-10, 20])
plt.grid(True)


points = []


def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        points.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'ro') 
        plt.draw()
def onkey(event):
    global points
    if event.key == "d":
        draw(points)
# Connect the click event to the handler function
cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid_key = fig.canvas.mpl_connect('key_press_event', onkey)


def draw(points):
    x=[]
    y=[]
    for x1,y1 in points:
        x.append(x1)
        y.append(y1)
    x2=[val**2 for val in x]
    #X = "1 + x + x^2"
    X=np.concatenate( (np.transpose([np.ones(len(points))]) , np.transpose([x]) , np.transpose([x2])), axis=1)
    #XTX = XT * X
    XTX=(np.matmul(np.transpose(X),X))
    # Inverting X^TX matrix
    XTXinv=np.linalg.inv(XTX)
    # ab = y * X * (XT * X)^-1
    abc=np.matmul(np.matmul(y,X),XTXinv)
    print(abc)
    a=round(abc[2],6)
    b=round(abc[1],6)
    c=round(abc[0],6)

    xaxis=np.linspace(-20,20)
    yaxis=a*xaxis**2+b*xaxis+c

    #ax.scatter(x,y,label = "Pisteet")
    ax.plot(xaxis,yaxis,label=f"y={a}x^2+{b}x+{c}")
    plt.legend()
    plt.grid(True)

    print(f"{abc[2]}x^2+{abc[1]}x + {abc[0]}")
    plt.show()
plt.show()