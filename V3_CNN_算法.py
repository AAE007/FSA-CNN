import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from CNN_model import Conv1DNet
from torchsummary import summary
import torch.nn as nn


# 从Excel文件中读取数据
excel_file = "./image.xlsx"
data_frame = pd.read_excel(excel_file).to_numpy()

# 划分训练集和测试集
X_train = data_frame[1:26, 1:6].astype(float)
y_train = data_frame[1:26, 6].astype(float)
y_train = y_train[:, np.newaxis]

X_test = data_frame[27:, 1:6].astype(float)
y_test = data_frame[27:, 6].astype(float)
y_test = y_test[:, np.newaxis]

# 定义超参数
input_size = 5
output_size = 1
learning_rate = 0.0001
num_epochs = 20000
batch_size = len(y_train)
channels_1 = 32
channels_2 = 64

# 创建数据集和数据加载器
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

print(X_train.size())

train_dataset = data.TensorDataset(X_train, y_train)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(X_test, y_test)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
net = Conv1DNet(input_size, output_size, channels_1, channels_2)


# 定义目标函数，这里以Rastrigin函数为例
# Rastrigin函数是一个多峰函数，具有很多局部最优解，全局最优解为f(0,0)=0
# 你可以根据你的问题修改这个函数

# 定义钩子函数,为打印激活热力图做准备
activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def get_loss(input_size, output_size, x):
    torch.manual_seed(99)  # 设置相同的随机种子
    fish_x, fish_y = x

    # # 使用最佳模型参数进行打印激活热力图
    # fish_x, fish_y = [36,68]

    # 训练模型
    net = Conv1DNet(input_size, output_size, fish_x, fish_y)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # # 遍历模型的子模块并输出各层的类型和可训练参数量
    # summary(net, (5,))

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

    # 评估模型
    net.eval()
    test_loss = 0.0

    for inputs, labels in test_loader:
        outputs = net(inputs)
        loss = net.compute_loss_t(outputs, labels)
        test_loss += loss.item()

    test_loss = test_loss / len(test_loader)

    #
    # # 注册钩子到目标层，例如ResNet的第一个卷积层
    # net.conv1d1.register_forward_hook(get_activation('conv1d1'))
    #
    # # 注册钩子到目标层，例如ResNet的第二个卷积层
    # net.conv1d2.register_forward_hook(get_activation('conv1d2'))
    #
    # # 准备输入数据并执行前向传播
    # input = torch.randn((25,5))
    # output = net(input)
    #
    # # 获取激活并处理
    # act_1 = activation['conv1d1'].squeeze()
    # # 例如，取平均值以减少通道维度
    # avg_act = torch.mean(act_1, dim=0)
    #
    # # print(avg_act.shape)
    # # 生成并显示热力图
    # plt.imshow(avg_act, cmap='coolwarm', interpolation='nearest')
    # plt.colorbar()
    # # 保存图像为SVG格式
    # plt.savefig("../paper/activation_heatmap_layer_1.svg", format='svg')
    # plt.show()
    #
    # # 矩阵打印为直方图，将矩阵展平为一维数组
    # flattened_matrix = avg_act.flatten()
    #
    # # 绘制直方图
    # plt.hist(flattened_matrix, bins=50, color="#549454")  # bins='auto' 让matplotlib自动选择合适数量的柱子
    # # 设置y轴范围
    # plt.ylim(0, 12)
    # # 设置y轴范围
    # plt.xlim(-1, 1)
    # ax = plt.gca()
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.tick_params(axis='both', length=0)
    # # plt.xlabel('Value')
    # # plt.ylabel('Frequency')
    # # 调整图边界
    # plt.tight_layout()
    #
    # # 进一步调整子图间距
    # plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    #
    # # 设置边框粗细为0.5
    # for spine in ax.spines.values():
    #     spine.set_linewidth(0.5)
    #
    # # 将尺寸从毫米转换为英寸
    # width_mm = 40
    # height_mm = 20
    # width_in = width_mm / 25.4
    # height_in = height_mm / 25.4
    #
    # # 设置图形尺寸
    # plt.gcf().set_size_inches(width_in, height_in)
    # plt.savefig("../paper/activation_heatmap_histogram_layer_1.png",transparent=True, format='png',dpi=1200)
    # plt.show()
    #
    # # 获取激活并处理
    # act_2 = activation['conv1d2'].squeeze()
    # # 例如，取平均值以减少通道维度
    # avg_act = torch.mean(act_2, dim=0)
    #
    # # print(avg_act.shape)
    # # 生成并显示热力图
    # plt.imshow(avg_act, cmap='coolwarm', interpolation='nearest')
    # plt.colorbar()
    # # 保存图像为SVG格式
    # plt.savefig("../paper/activation_heatmap_layer_2.svg", format='svg')
    # plt.show()
    #
    # # 矩阵打印为直方图，将矩阵展平为一维数组
    # flattened_matrix = avg_act.flatten()
    #
    # # 绘制直方图
    # plt.hist(flattened_matrix, bins=50, color="#549454")  # bins='auto' 让matplotlib自动选择合适数量的柱子
    # # 设置y轴范围
    # plt.ylim(0, 16)
    # # 设置y轴范围
    # plt.xlim(-1, 1)
    # ax = plt.gca()
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.tick_params(axis='both', length=0)
    # # plt.xlabel('Value')
    # # plt.ylabel('Frequency')
    # # 调整图边界
    # plt.tight_layout()
    #
    # # 进一步调整子图间距
    # plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    #
    # # 设置边框粗细为0.5
    # for spine in ax.spines.values():
    #     spine.set_linewidth(0.5)
    #
    # # 将尺寸从毫米转换为英寸
    # width_mm = 40
    # height_mm = 20
    # width_in = width_mm / 25.4
    # height_in = height_mm / 25.4

    # # 设置图形尺寸
    # plt.gcf().set_size_inches(width_in, height_in)
    # plt.savefig("../paper/activation_heatmap_histogram_layer_2.png",transparent=True, format='png', dpi=1200)
    # plt.show()

    return test_loss


# 定义鱼类，每条鱼有两个属性：位置和适应度
class Fish:
    def __init__(self, x):
        self.position = x  # 位置是一个包含两个元素的数组，表示鱼在参数空间中的坐标
        self.fitness = get_loss(input_size, output_size, x)  # 适应度是目标函数在该位置的值，表示鱼的食物浓度


# 定义鱼群类，包含多条鱼和一些算法参数
class FishSwarm:
    def __init__(self, n_fish, n_iter, visual, step, delta):
        self.n_fish = n_fish  # 鱼的数量
        self.n_iter = n_iter  # 迭代次数
        self.visual = visual  # 鱼的视野范围，表示鱼能感知的最远距离
        self.step = step  # 鱼的步长，表示鱼每次移动的最大距离
        self.delta = delta  # 鱼的拥挤度因子，表示鱼之间的最小距离
        self.fishes = []  # 鱼的列表，存储每条鱼的对象
        self.best_fish = None  # 最优鱼，存储全局最优解的鱼的对象
        self.best_fitness = np.inf  # 最优适应度，存储全局最优解的目标函数值
        self.best_position = None  # 最优位置，存储全局最优解的参数值
        self.fitness_curve = []  # 适应度曲线，存储每次迭代后的最优适应度

        # 初始化鱼群，随机生成每条鱼的初始位置和适应度
        for i in range(n_fish):
            x = np.random.randint(5, 128, 2)  # 在[-5.12, 5.12]区间内随机生成两个参数值
            print(x)
            fish = Fish(x)  # 创建鱼对象
            self.fishes.append(fish)  # 将鱼对象添加到鱼列表中
            # 更新最优鱼，最优适应度和最优位置
            if fish.fitness < self.best_fitness:
                self.best_fish = fish
                self.best_fitness = fish.fitness
                self.best_position = fish.position

    # 定义鱼的觅食行为，表示鱼向当前邻域内食物浓度更高的位置移动
    def forage(self, input_size, output_size, fish):
        # 生成一个新的位置，表示鱼在视野范围内随机移动一步
        new_position = fish.position + np.random.randint(-1, 1, 2) * self.step
        new_position = np.round(new_position).astype(int)
        for i in range(len(new_position)):
            if new_position[i] <= 0:
                new_position[i] += 5
        # 计算新位置的适应度，表示新位置的食物浓度
        new_fitness = get_loss(input_size, output_size, new_position)
        # 如果新位置的适应度小于当前位置的适应度，表示找到了更好的位置，就更新鱼的位置和适应度
        if new_fitness < fish.fitness:
            fish.position = new_position
            fish.fitness = new_fitness
            # 如果新位置的适应度小于最优适应度，表示找到了全局最优解，就更新最优鱼，最优适应度和最优位置
            if new_fitness < self.best_fitness:
                self.best_fish = fish
                self.best_fitness = new_fitness
                self.best_position = new_position

    # 定义鱼的聚群行为，表示鱼向当前邻域内鱼的中心位置移动
    def swarm(self, input_size, output_size, fish):
        # 初始化邻域内的鱼的数量，中心位置，平均适应度
        n_local = 0
        center = np.zeros(2)
        mean_fitness = 0
        # 遍历鱼群中的每条鱼
        for other in self.fishes:
            # 如果其他鱼与当前鱼的距离小于视野范围，表示在邻域内
            if np.linalg.norm(fish.position - other.position) < self.visual:
                # 更新邻域内的鱼的数量，中心位置，平均适应度
                n_local += 1
                center += other.position
                mean_fitness += other.fitness
        # 计算邻域内的鱼的中心位置，平均适应度
        center /= n_local
        mean_fitness /= n_local
        # 如果邻域内的平均适应度小于当前鱼的适应度，表示邻域内有更好的位置
        if mean_fitness < fish.fitness:
            # 生成一个新的位置，表示鱼向邻域内的中心位置移动一步
            new_position = fish.position + (center - fish.position) * np.random.randint(0,
                                                                                        1) * self.step / np.linalg.norm(
                center - fish.position)
            new_position = np.round(new_position).astype(int)
            for i in range(len(new_position)):
                if new_position[i] <= 0:
                    new_position[i] += 5
            # 计算新位置的适应度，表示新位置的食物浓度
            new_fitness = get_loss(input_size, output_size, new_position)
            # 如果新位置的适应度小于当前位置的适应度，表示找到了更好的位置，就更新鱼的位置和适应度
            if new_fitness < fish.fitness:
                fish.position = new_position
                fish.fitness = new_fitness
                # 如果新位置的适应度小于最优适应度，表示找到了全局最优解，就更新最优鱼，最优适应度和最优位置
                if new_fitness < self.best_fitness:
                    self.best_fish = fish
                    self.best_fitness = new_fitness
                    self.best_position = new_position

    # 定义鱼的追尾行为，表示鱼向当前邻域内适应度最好的鱼移动
    def follow(self, input_size, output_size, fish):
        # 初始化邻域内的鱼的数量，最优适应度，最优位置
        n_local = 0
        best_fitness = np.inf
        best_position = None
        # 遍历鱼群中的每条鱼
        for other in self.fishes:
            # 如果其他鱼与当前鱼的距离小于视野范围，表示在邻域内
            if np.linalg.norm(fish.position - other.position) < self.visual:
                # 更新邻域内的鱼的数量
                n_local += 1
                # 如果其他鱼的适应度小于邻域内的最优适应度，表示找到了邻域内的最优鱼
                if other.fitness < best_fitness:
                    # 更新邻域内的最优适应度，最优位置
                    best_fitness = other.fitness
                    best_position = other.position
                # 如果邻域内的最优适应度小于当前鱼的适应度，就更新鱼的位置和适应度
                if best_position is not None and best_fitness < fish.fitness:
                    # 生成一个新的位置，表示鱼向邻域内的最优鱼移动一步
                    new_position = fish.position + (best_position - fish.position) * np.random.randint(0,
                                                                                                       1) * self.step / np.linalg.norm(
                        best_position - fish.position)
                    new_position = np.round(new_position).astype(int)
                    for i in range(len(new_position)):
                        if new_position[i] <= 0:
                            new_position[i] += 5
                    # 计算新位置的适应度，表示新位置的食物浓度
                    new_fitness = get_loss(input_size, output_size, new_position)
                    # 如果新位置的适应度小于当前位置的适应度，表示找到了更好的位置，就更新鱼的位置和适应度
                    if new_fitness < fish.fitness:
                        fish.position = new_position
                        fish.fitness = new_fitness
                        # 如果新位置的适应度小于最优适应度，表示找到了全局最优解，就更新最优鱼，最优适应度和最优位置
                        if new_fitness < self.best_fitness:
                            self.best_fish = fish
                            self.best_fitness = new_fitness
                            self.best_position = new_position

    # 定义鱼群算法的主要迭代过程
    def optimize(self):
        # 迭代指定的次数
        for _ in range(self.n_iter):
            # 对鱼群中的每条鱼执行觅食，聚群和追尾行为
            for fish in self.fishes:
                self.forage(input_size, output_size, fish)
                self.swarm(input_size, output_size, fish)
                self.follow(input_size, output_size, fish)

            print(self.best_position, self.best_fitness)
            # 记录当前迭代后的最优适应度
            self.fitness_curve.append(self.best_fitness)

        print(self.best_position)
        # 绘制适应度曲线
        plt.plot(self.fitness_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness Curve of Fish Swarm Optimization')
        plt.show()


# 创建鱼群对象，设置参数值
swarm = FishSwarm(n_fish=10, n_iter=50, visual=5, step=1, delta=0.1)
# 执行鱼群算法的优化过程
swarm.optimize()
