# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020-10-20 17:42
# 文件名称：generate_train
# 开发工具：PyCharm

import os
x_train_path = '../dataset2/jpg'
# 得到所有x名字
x_train_name = os.listdir(x_train_path)


y_train_path = '../dataset2/png'
# 得到所有y名字
y_train_name = os.listdir(y_train_path)


# 写入train.txt
train_txt_path = '../dataset2/train1.txt'
txt = open(train_txt_path,"w")

for i in range(len(x_train_name)):
    txt.write(x_train_name[i]+";"+y_train_name[i]+"\n")
txt.close()

