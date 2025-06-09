import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle

def main():
    base_dir = os.path.dirname(__file__)
    # Load CIFAR-10 dataset
    train_loader, test_loader = load_data(100, base_dir)
    
    # Check if the model already exists
    model_path = os.path.join(base_dir, 'cifar10-cnn-model.pth')
    model_exists = True if os.path.exists(model_path) else False
    print(f"Model exists: {model_exists}")
    # Create CNN model ( Convolutional Neural Network )
    model = create_cnn()
    if model_exists:
        # loading model
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
    else:
        # Train the model
        lr = 0.001
        epochs = 20 
        model = train_model(model, train_loader, test_loader, epochs=epochs, learning_rate=lr)
        save_model(model, path = model_path)

    test_model(model, test_loader)
    if not model_exists:
        save_model(model, path='cifar10-cnn-model.pth')

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def load_data(batch_size, base_dir):
    # Define the path to the CIFAR-10 dataset
    data_dir = os.path.join(base_dir, 'cifar-10-batches-py')
    
    # Load training data
    train_data = []
    train_labels = []
    # there are 5 data_batches in CIFAR-10
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    # Combine batches into single array
    train_data = np.concatenate(train_data)
    train_data = train_data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # Dividing by 255.0 so RGB values are in range [0,1]
    train_labels = np.array(train_labels)
    
    # Load test data
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_labels = np.array(test_batch[b'labels'])
    
    # Create DataLoader for training and testing data
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
def create_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3 channels (RGB), Output: 32 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2 now image size is 16x16
        nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input: 32 channels, Output: 64 channels
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample by a factor of 2 now image size is 8x8
        nn.Flatten(),  # Flatten the output for the fully connected layer making the size 64 * 8 * 8
        nn.Linear(64 * 8 * 8, 512),  # Fully connected layer
        nn.ReLU(),
        nn.Linear(512, 10)  # Output layer for 10 classes
    )
    return model
def train_model(model, train_loader, test_loader, epochs, learning_rate):
    device = torch.device('cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss() # Loss function is cross entropy for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad() # Always zero the gradients to start fresh
            outputs = model(images)
            loss = criterion(outputs, labels) # Cross entropy loss between the predicted and true labels
            loss.backward() # Backpropagation of the loss function with respect to the model parameters
            optimizer.step() # Updating the model parameters based on the gradients
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    print('Training complete.')
    
    return model
def test_model(model, test_loader):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
    class_correct = [0] * 10
    class_total = [0] * 10
    device = torch.device('cpu')
    model.to(device)
    # Testing the model
    model.eval()
    correct = 0
    total = 0
    #tested = False
    with torch.no_grad(): # Grads arent needed for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the max probability class for each image
            total += labels.size(0)
            for i in range(len(labels)):
                class_correct[predicted[i]] += 1 if predicted[i] == labels[i] else 0
                class_total[labels[i]] += 1
            '''if not tested: If u want to see the predicted and true classes
                i = 0
                for val in predicted:
                    print(f"predicted class: {classes[val]}, true class: {classes[labels[i]]}")
                    i += 1
                tested = True'''
            correct += (predicted == labels).sum().item()
    print(f"Amount of images: {total}, Correctly guessed images {correct}")
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')




if __name__ == '__main__':
    main()