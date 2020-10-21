# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020-10-20 21:12
# 文件名称：mypredict
# 开发工具：PyCharm

from PIL import Image
import numpy as np
from nets.segnet import mobilenet_segnet

classes_num = 2
model_weights_path = 'logs/ep003-loss0.050-val_loss0.144.h5'
colors = [[0, 0, 0], [0, 0, 255]]


# img = input("输入图片的路径：")


# 对输入图片进行出处理
def predict(image_path):
    # 先读入图像
    image = Image.open(image_path)
    # 对图像resize到416,416,3
    image = image.resize((416, 416))
    image_copy = image.copy()
    image_copy = np.array(image_copy)
    # 图像转为array
    image = np.array(image)
    # 图像归一化
    image = image / 255
    # 弄成批量
    image = image.reshape(1, 416, 416, 3)

    # 初始化模型
    model = mobilenet_segnet(classes_num, 416, 416)
    # 模型加载权重
    model.load_weights(model_weights_path)

    # 进行预测，result = (43264,2)
    result = model.predict(image)[0]
    # (43264,2->208,208,2)
    result = result.reshape(208, 208, classes_num)
    # (208,208,2)->(208,208)
    result = result.argmax(axis=-1)

    seg_img = np.zeros((208, 208, 3))
    for c in range(classes_num):
        seg_img[:, :, 0] += ((result == c) * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((result == c) * colors[c][1]).astype('uint8')
        seg_img[:, :, 1] += ((result == c) * colors[c][2]).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((image_copy.shape[0], image_copy.shape[1]))

    image_copy = Image.fromarray(image_copy)
    image = Image.blend(image_copy, seg_img, 0.5)
    image.show()


# image_path = input("输入图片路径")
predict("dataset2/jpg/1.jpg")
