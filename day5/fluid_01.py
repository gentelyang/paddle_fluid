# -*- coding=utf-8
import numpy as np
import paddle.fluid as fluid
a=fluid.layers.data(name="a",shape=[1],dtype="float32")
b=fluid.layers.data(name="b",shape=[1],dtype="float32")
result=fluid.layers.elementwise_add(a,b)

cpu=fluid.core.CPUPlace()#定义运算的场所，这里选择cpu
exe=fluid.Executor(cpu)#创建执行器，是大写的Executodr，不是小写的。
exe.run(fluid.default_startup_program())#网络参数初始化

data_1=int(input("Please enter an integer: a="))
data_2=int(input("Please enter an integer: b="))
x=np.array([[data_1]])
y=np.array([[data_2]])

outs=exe.run(feed={'a':x,'b':y},
             fetch_list=[result.name])
print "%d+%d=%d" % (data_1,data_2,outs[0][0])
print "the number of i is"

