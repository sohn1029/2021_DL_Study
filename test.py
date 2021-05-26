import sys, os
import pandas as pd
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from convnet import SimpleConvNet
from utils.layer import *
from utils.function import numerical_gradient
import matplotlib.pyplot as plt
import datetime
import cupy as cp

d = datetime.datetime.now()
formatted_d = "{0}_{1}_{2}_{3}_{4}".format(d.month, d.day, d.hour, d.minute, d.second)

with open("./models/params_5_26_21_21_26.pkl", 'rb') as file : 
    w = []
    while True:
        try:
            w_data = pickle.load(file)
        except EOFError : 
            break
        w.append(w_data)

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=260, output_size=26, weight_init_std=0.01, params = w[0])



test_data = pd.read_csv("./handwritten_alphabet_dataset/test_set.csv")
x_test, t_test = test_data.to_numpy()[:,1:].reshape(test_data.shape[0],1,28,28), test_data.to_numpy()[:,0]

x_test, t_test = x_test[:100], t_test[:100]
x_test = cp.array(x_test)
#t_test = cp.array(t_test)

y_hat = network.predict(x_test)
y_hat = cp.asnumpy(y_hat)

tmp = []
for i in range(100):
    tmp.append(np.where(y_hat[i] == y_hat[i].max())[0][0])

acc = np.count_nonzero(t_test == np.array(tmp)) / 100

x_test = cp.asnumpy(x_test)

fig = plt.figure()
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.xlabel(chr(65 + np.where(y_hat[i] == y_hat[i].max())[0][0]))
plt.title("Acc : {0}".format(acc))
plt.savefig("predict_{0}.png".format(formatted_d))



print(acc)

