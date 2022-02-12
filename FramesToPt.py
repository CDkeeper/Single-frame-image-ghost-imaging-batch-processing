import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2 as cv
import torchvision.transforms
from torchvision.transforms import functional as F
image_size = 64  # 128
iteration = 128  # 实验室电脑内存极限大致是 450*60000*28*28
batch_ci = 5
batch_size = 2

image = np.zeros((batch_size, iteration, image_size, image_size))  # 设置图片数组
transform = torchvision.transforms.ToTensor()
pre_data = torch.load(r"C:\Users\chenda\Desktop\MNist\train-pt/1.pt")
# 读取图片
for num in range(batch_ci):
    torch.cuda.empty_cache()

    # 初始化 B 数组
    B = torch.zeros((iteration, batch_size)).cuda()
    BI = torch.zeros((batch_size, iteration, image_size, image_size)).cuda()

    # [500, 28, 28]的图像数组
    img_batch = torch.zeros((batch_size, image_size, image_size)).cuda()
    img_batch = img_batch.cuda()
    for i in range(batch_size):
        img_batch[i] = pre_data[i + batch_size * num].cuda()
        # # [512]
    sum_I1 = torch.zeros(batch_size).cuda()
    # [512, 128, 128]
    sum_I2 = torch.zeros(img_batch.shape).cuda()
    # [512, 128, 128]
    sum_ans = torch.zeros(img_batch.shape).cuda()

    sum_I2 = sum_I2.permute(1, 2, 0)
    sum_ans = sum_ans.permute(1, 2, 0)

    field = np.random.normal(0, 1, [iteration, image_size, image_size])
    field = torch.from_numpy(field).cuda()
    for k in range(iteration):
        temp = field[k].repeat(batch_size, 1, 1).cuda()  # 每次用一样的图案照射本批次的所有物体。
        B[k] = torch.mul(temp, img_batch).sum(2).sum(1)
    B = B.permute(1, 0).cpu()  # 批数在前，帧数在后

    for i in range(batch_size):
        for j in range(iteration):
            BI[i][j] = B[i][j] * field[j]
    torch.save(BI, r"C:\Users\chenda\Desktop\MNist\train-GI-pt/{}.pt".format(num + 1))  # 数据的保存路径

    #     # 桶测量值求和
    #     temp = temp.permute(1, 2, 0)
    #     sum_I1 = sum_I1 + B[k]
    #     # 热光矩阵求和
    #     sum_I2 = sum_I2 + temp
    #     # 桶测量值乘热光矩阵求和
    #     sum_ans = sum_ans + temp * B[k]
    # sum_I1 = sum_I1 / iteration
    # sum_I2 = sum_I2 / iteration
    # sum_ans = sum_ans / iteration
    # # [512, 128, 128]
    # ans = sum_ans - sum_I1 * sum_I2
    # tmpres = torch.zeros((batch_size, image_size, image_size)).cuda()
    # for i in range(batch_size):
    #     res = ans[:, :, i]
    #     mx = res.max()
    #     mi = res.min()
    #     res = 255 * (res - mi) / (mx - mi)
    #     tmpres[i] = res
    # print(tmpres.shape)
    # torch.save(tmpres, r"C:\Users\chenda\Desktop\MNist\ceshi/{}.pt".format(num + 1))  # 数据的保存路径