import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 加载训练集和测试集数据
train_data = pd.read_excel("complex_nonlinear_data.xlsx")
test_data = pd.read_excel("new_complex_nonlinear_data.xlsx")

# 获取训练集和测试集的输入和输出数据
X_train, y_train = train_data["x"].values, train_data["y_complex"].values
X_test, y_test = test_data["x_new"].values, test_data["y_new_complex"].values

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建神经网络模型实例
model = NeuralNetwork()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练神经网络模型
num_epochs = 50000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

# 测试神经网络模型
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    mse = test_loss.item()

print("MSE:", mse)

# 可视化训练过程中的损失值变化
x_plot = np.linspace(0, 10, 1000)
x_plot_tensor = torch.tensor(x_plot, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    y_plot_tensor = model(x_plot_tensor)
y_plot = y_plot_tensor.numpy()

plt.scatter(X_train, y_train, label="Training Data")
plt.plot(x_plot, y_plot, color="red", label="Fitted Curve")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Fitted Curve vs. Training Data")
plt.legend()
plt.show()