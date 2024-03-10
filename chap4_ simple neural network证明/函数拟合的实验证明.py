# 2150241 赵彦平
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 要拟合的函数
def custom_function(x):
    return x*x + x*3 + 5

class ReLUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ReLUNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_network(model, x_train, y_train, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        inputs = x_train
        targets = y_train
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    print('Training finished.')

x_train = torch.unsqueeze(torch.linspace(-10,10,100), dim=1)
y_train = custom_function(x_train)

input_size = 1
hidden_size = 100
output_size = 1
num_epochs = 20000
learning_rate = 0.0009

model = ReLUNetwork(input_size, hidden_size, output_size)

train_network(model, x_train, y_train, num_epochs, learning_rate)

x_test = torch.unsqueeze(torch.linspace(-10,10,100), dim=1)
y_test = custom_function(x_test)

model.eval()
with torch.no_grad():
    outputs = model(x_test)

predicted = outputs.numpy()

plt.plot(x_test, y_test, label='Original Function')
plt.plot(x_test, predicted, label='Predicted Function')
plt.legend()
plt.show()
