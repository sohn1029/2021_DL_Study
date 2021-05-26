# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
#from dataset.mnist import load_mnist
from convnet import SimpleConvNet
from common.trainer import Trainer
import pandas as pd
import torch
import datetime

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 시간이 오래 걸릴 경우 데이터를 줄인다.
# x_train, t_train = x_train[:5000], t_train[:5000]
# x_test, t_test = x_test[:1000], t_test[:1000]


# Handwritten Alphabet data
trainset = pd.read_csv("./handwritten_alphabet_dataset/train_set.csv")
validset = pd.read_csv("./handwritten_alphabet_dataset/valid_set.csv")

train_y = np.array(trainset["0"])[:20000] # label
train_x = np.array(trainset.drop("0", axis=1))[:20000].reshape(-1, 1, 28, 28) # (238368, 1, 28, 28)
# train_x = np.array([deskew(img) for img in train_x.reshape(-1, 784)])

valid_y = np.array(validset["0"])[:2000] # label
valid_x = np.array(validset.drop("0", axis=1))[:2000].reshape(-1, 1, 28, 28) # (59592, 1, 28, 28)

print("trainset : {0}, validset : {1}".format(train_x.shape, valid_x.shape))

max_epochs = 50

model = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=260, output_size=26, weight_init_std=0.01)
                        
trainer = Trainer(model, train_x, train_y, valid_x, valid_y,
                  epochs=max_epochs, mini_batch_size=200,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 매개변수 보존
d = datetime.datetime.now()
formatted_d = "{0}_{1}_{2}_{3}_{4}".format(d.month, d.day, d.hour, d.minute, d.second)
model.save_params("./models/params_{0}.pkl".format(formatted_d))
print("Saved Network Parameters!")

# 그래프 그리기
markers = {'train': 'o', 'valid': 's'}
x = np.arange(len(trainer.train_acc_list))
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='valid', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.savefig("./results/acc_{0}.png".format(formatted_d))
plt.show()