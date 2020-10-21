# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020-10-20 19:13
# 文件名称：predict
# 开发工具：PyCharm
import os
from PIL import Image
import copy
import numpy as np

from nets.segnet import mobilenet_segnet

class_colors = [[0, 0, 0], [0, 0, 255]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416
model = mobilenet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)
path='logs/ep003-loss0.050-val_loss0.144.h5'
model.load_weights(path)

# 展示dataset2/jpg下面所有图片名字
imgs = os.listdir("dataset2/jpg/")
np.random.shuffle(imgs)
i = 0
for jpg in imgs:
    if i>20:
        break
    img = Image.open("./dataset2/jpg/" + jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    # 将图像大小设为416,416,3
    img = img.resize((WIDTH, HEIGHT))
    img = np.array(img)
    img = img / 255
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    # 得到预测结果43264，2
    pr = model.predict(img)[0]
    # （43264,2）-->(208,208,2)
    pr = pr.reshape((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), 3))
    colors = class_colors
    for c in range(NCLASSES):
        seg_img[:, :, 0] = seg_img[:, :, 0] + ((pr[:, :] == c) * colors[c][0]).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

    image = Image.blend(old_img, seg_img, 0.3)
    image.save("./img_out/" + jpg)
    i = i+1
