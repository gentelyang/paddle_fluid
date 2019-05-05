# -*- coding=utf-8
import numpy as np
import paddle.fluid as fluid

# 定义数据
train_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
y_true = np.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')

# 定义网络结构
# 首先定义输入数据模型
x = fluid.layers.data(name="x", shape=[1], dtype='float32')
y = fluid.layers.data(name="x", shape=[1], dtype='float32')
# 再定义全链接层模型
y_predict = fluid.layers.fc(input=x, size=1, act=None)

# 模型搭建完成后需要评估模型好坏，通过设计损失函数计算 真实值和预测值之间的差
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

# 网络结构和数据准备好以后开始训练
# outs=exe.run(feed={'x':train_data},
#         fetch_list=[y_predict,avg_cost])


# 定义损失函数后，可以通过前向计算得到损失值，通过链式求导法则计算梯度。获取梯度后更新参数
# 定义优化方法
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)
cpu = fluid.core.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())
# 开始训练，迭代100次
for i in range(100):
    outs = exe.run(feed={'x': train_data},
                   fetch_list=[y_predict, avg_cost])
print type(outs)
print outs
