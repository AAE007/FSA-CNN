import torch
import torch.nn as nn


# 定义模型
class Conv1DNet(nn.Module):
    def __init__(self, input_size, output_size, channels_1, channels_2):
        super(Conv1DNet, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=1, out_channels=channels_1, kernel_size=2)
        self.conv1d2 = nn.Conv1d(in_channels=channels_1, out_channels=channels_2, kernel_size=2)
        self.fc1 = nn.Linear(channels_2 * 3, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1d1(x.unsqueeze(1))
        x = self.relu(x)
        x = self.conv1d2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return x

    def compute_loss(self, output, target):
        loss = torch.linalg.norm(output - target, ord=1)
        return loss

    def compute_loss_t(self, output, target):
        self.criterion_t = nn.MSELoss()
        loss = self.criterion_t(output, target)
        return loss

