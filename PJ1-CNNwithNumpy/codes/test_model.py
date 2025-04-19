import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

#如果要使用MLP，则将注释部分激活
# model = nn.models.Model_MLP()
model=nn.models.Model_CNN()
model.load_model(r'.\saved_models\best_model_CNNwithpoolanddropout0_2.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

# # 如果要使用MLP，则将下面注释部分激活，并将CNN注释
# with gzip.open(test_images_path, 'rb') as f:
#         magic, num, rows, cols = unpack('>4I', f.read(16))
#         test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

# 加载图像数据 CNN:
with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)  # shape (60000, 28, 28)
    test_imgs = test_imgs[:, np.newaxis, :, :]  # 最终shape (60000, 1, 28, 28)

with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

model.eval()
logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))