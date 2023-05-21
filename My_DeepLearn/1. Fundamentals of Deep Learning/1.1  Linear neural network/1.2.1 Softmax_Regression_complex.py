import torch
from IPython import display
from d2l import torch as d2l
from d2l.mxnet import Animator

if __name__ == '__main__':

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 将平展每个图像，将他们视为长度为784的向量。因为我们的数据集有10个类别，所以网络输出维度为10
    num_inputs = 784
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 实现softmax
    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        # 这里应用了广播机制(第i行除以第i个元素)
        return X_exp / partition

    # 实现softmax模型
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    # 创建一个数据y_hat,其中包含2个样本在3个类别的预测概率，使用y作为y_hat中概率的索引
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

    # 实现交叉熵损失函数
    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    # 将预测类别与真实y元素进行比较
    def accuracy(y_hat, y):
        """计算预测正确的数量"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    # print(f'accuracy: {accuracy(y_hat, y) / len(y)}')

    class Accumulator:
        def __init__(self, n):
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # 评估在任意模型net的准确度
    def evaluate_accuracy(net, data_iter):
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval() # 将模型设置为评估模式
        metirc = Accumulator(2) # 正确预测数、预测总数
        for X, y in data_iter:
            metirc.add(accuracy(net(X), y), y.numel())
        return metirc[0] / metirc[1]

    print(evaluate_accuracy(net, test_iter))

    # 训练
    def train_epoch_ch3(net, train_iter, loss, updater):
        """训练模型一个迭代周期"""
        # 将模型设置为训练模式
        if isinstance(net, torch.nn.Module):
            net.train()
        # 训练损失总和、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in train_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    # 训练函数
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
        """训练模型（定义见第3章）"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    # 小批量随机梯度下降来优化模型的损失函数
    lr = 0.1

    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)

    # 训练模型10个迭代周期
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()

    # 对图像进行分类预测
    def predict_ch3(net, test_iter, n=6):  #@save
        """预测标签"""
        for X, y in test_iter:
            break
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(
            X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

    predict_ch3(net, test_iter)

