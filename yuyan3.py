import numpy as np
import cv2

# 将镜头参数代入相机内参矩阵和畸变向量中
fx = ...  # 焦距
fy = ...  # 焦距
cx = ...  # 光学中心的 x 坐标
cy = ...  # 光学中心的 y 坐标
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
k1 = ...  # 畸变参数 k1
k2 = ...  # 畸变参数 k2
p1 = ...  # 畸变参数 p1
p2 = ...  # 畸变参数 p2
d = np.array([k1, k2, p1, p2])

# 加载需要进行畸变校正的图像
img = cv2.imread("example.jpg")

# 进行畸变校正
img_undistorted = cv2.undistort(img, K, d)

# 处理畸变校正后的图像，例如进行后续的图像处理或者分析
