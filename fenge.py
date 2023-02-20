import cv2
import numpy as np

# 读取图像
img = cv2.imread('data/53.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150)

# 膨胀边缘
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# 进行轮廓检测
contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 查找最大轮廓
max_contour = max(contours, key=cv2.contourArea)

# 绘制最大轮廓
cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 3)

# 显示结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
