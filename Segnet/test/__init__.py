# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020-10-19 20:28
# 文件名称：__init__.py
# 开发工具：PyCharm


import numpy as np
from keras import Model
from keras.layers import *
import keras.backend as K
from PIL import Image

# w = np.array([5, 6, 7, 8, -1])
# w = w.reshape(1,5)
# x = Input(shape=(5,))
# z = Activation("relu")(x)
# model = Model(x,z)
# print(model.predict(w))
# print("哈")


# x = np.array([[1,2,3],[4,5,6]])
# x = x.reshape(1,2,3,1)
# input = Input(shape=(2,3,1))
# out = ZeroPadding2D(padding=((1,1),(1,0)))(input) # 上下各加1行，左加1行，右加一行
# model = Model(input,out)
# print(model.predict(x).reshape(4,4))

# x = np.array([1,2,3,4,5,6])
# x = x.reshape(1,6)
# def relu6(y):
#     return K.relu(y, max_value=4)
# input = Input(shape=(6,))
# out = Activation(relu6)(input)
# model = Model(input,out)
# model.summary()
# print(model.predict(x))


# img = Image.open("../dataset2\png"+"/1.png")
# img = img.resize((208,208))
# img = np.array(img)
# print(img.shape)


# x = "love"
# x = x.replace("lo","hh") # 将lo替换为hh
# print(x)


# x = np.zeros((3,3,3))
# x[:,:,0] = 1
# print(x.shape)
# print(x[0])

# x = np.zeros((2,2))
# x[:,0] = 1
# print(x)


# img = Image.open("../dataset2\png" + "/1.png")
# img = img.resize((208,208))
# img = np.array(img)
# print(img[:,:,0])

x = np.array([1,2,3])
y = np.array([4,5,7])
print((x==1)*y)

