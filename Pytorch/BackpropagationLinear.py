import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Making synthetic data for regression froim sklearn
X_np, Y_np = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state= 1)

# Convert numpy arrays to pytorch tensors
X = torch.from_numpy(X_np.astype(np.float32))
Y = torch.from_numpy(Y_np.astype(np.float32))
Y = Y.view(Y.shape[0],1)

n_samples , n_features = X.shape

input_size = n_features
output_size = 1

model = nn.Linear(input_size,output_size)

learning_rate = 0.01
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr= learning_rate)

epochs = 100
for epoch in range(epochs):
    y_pred = model(X)
    # Calculating the loss using Mean Squared Error. (SUM of (y_pred - Y)^2 / n_samples)
    loss = loss_function(y_pred,Y)
    # Backpropagation of the loss function with respect to the model parameters
    loss.backward()
    # Updating the a and b parameters based on the gradients
    optimizer.step()
    # Zeroing the gradients for the next iteration
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")
predicted = model(X).detach().numpy()
# Plotting using matplotlib
plt.title('Linear Regression with Pytorch backpropagation')
plt.plot(X_np, Y_np,'ro')
plt.plot(X_np, predicted,'b')    
plt.show()