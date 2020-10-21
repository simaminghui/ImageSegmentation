# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020-10-20 15:58
# 文件名称：train
# 开发工具：PyCharm
import numpy as np
from PIL import Image
import keras.backend as K
from nets.segnet import mobilenet_segnet
from keras.callbacks import *
from keras.optimizers import *

NCLASSES = 2
HEIGHT = 416
WIDTH = 416


'''
（1）x:
x：把图像resize到416,416,3
（2）y:
①先把图像resize到208,208,3
②创建一个seg_labels,shape=(208,208,2)
③
'''

def generate_arrays_from_file(lines, batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while True:
        x_train = []
        y_train = []

        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i == 0:
                np.random.shuffle(lines)
            # 按照;划分，选取第一个.第一个是x，第二个是y
            name = lines[i].split(";")[0]

            # 从文件中读取图像
            img = Image.open(r"dataset2\jpg" + "/" + name)
            # 大小为416,416
            img = img.resize((WIDTH, HEIGHT))
            # shape = (416,416,3)
            img = np.array(img)
            # 归一化
            img = img / 255
            x_train.append(img)

            # 替换
            name = (lines[i].split(';')[1]).replace("\n", "")
            # 读取标签图像
            img_label = Image.open("dataset2\png" + "/" + name)
            img_label = img_label.resize((int(WIDTH / 2), int(HEIGHT / 2)))
            # 208,208,3
            img_label = np.array(img_label)
            # seg_labels.shape=(208,208,NCLASSES),全是0
            seg_labels = np.zeros((int(HEIGHT / 2), int(WIDTH / 2), NCLASSES))
            for c in range(NCLASSES):
                # 渠道
                seg_labels[:, :, c] = (img_label[:, :, 0] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))

            y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i + 1) % n
        yield (np.array(x_train), np.array(y_train))


def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true, y_pred)
    return loss


if __name__ == '__main__':
    log_dir = 'logs/'

    # 获得模型
    model = mobilenet_segnet(n_classes=NCLASSES, input_height=HEIGHT, input_width=WIDTH)

    # 获得权重
    # 刚开始的
    # weight_path = 'logs/mobilenet_1_0_224_tf_no_top.h5'

    # 经过训练的
    weight_path = 'logs/ep003-loss0.083-val_loss0.581.h5'
    model.load_weights(weight_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open("dataset2/train.txt", 'r') as f:
        lines = f.readlines()

    # 打乱行，这个txt朱勇用来帮助读取数据来训练
    # 打乱的数据更利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90% 用于训练，10%用于估计
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 保存方式，3世代保存一次
    # 监控val_loss
    checkpoint_period = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        save_best_only=True,
                                        period=3)

    # 学习率下降的方式，val_loss如果3次不下降，就改变学习率继续训练,学习率为原来的一半，0.5
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # 是否需要早听，当val_loss10次不下降的时候就停止训练,min_delta=0,表示么有变化
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # 交叉熵
    model.compile(loss=loss,
                  optimizer=Adam(1e-4),
                  metrics=['accuracy'])

    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period,reduce_lr,early_stopping]
                        )

    model.save_weights(log_dir+'last1.h5')
