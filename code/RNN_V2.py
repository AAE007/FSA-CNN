import torch
import torch.nn as nn

# 定义模型
class RNN_2(nn.Module):
    def __init__(self, input_size, output_size, channels_1, channels_2):
        super(RNN_2, self).__init__()
        self.rnn_1 = nn.RNN(input_size=input_size, hidden_size=channels_1, batch_first=True, num_layers = 1)
        self.rnn_2 = nn.RNN(input_size=channels_1, hidden_size=channels_2, batch_first=True, num_layers = 1)
        self.fc = nn.Linear(channels_2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 假设 x 的形状是 (batch_size, seq_length, input_size)
        x, _ = self.rnn_1(x.unsqueeze(1))
        x, _ = self.rnn_2(x)

        # 选择RNN输出的最后一个时间步
        x = x[:, -1, :]
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return x

    def compute_loss(self, output, target):
        loss = torch.linalg.norm(output - target, ord=1)
        return loss

    def compute_loss_t(self, output, target):
        self.criterion_t = nn.MSELoss()
        loss = self.criterion_t(output, target)
        return loss

