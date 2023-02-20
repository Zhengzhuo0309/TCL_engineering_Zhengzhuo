import cv2
import numpy as np

# 读取图像
img = cv2.imread('data/53.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用Canny算子进行边缘检测
edges = cv2.Canny(gray, 50, 150)

# 进行霍夫直线变换
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# 创建空白掩膜
mask = np.zeros_like(gray)

# 绘制直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

# 进行形态学操作，填充直线
kernel = np.ones((5, 5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 进行轮廓检测
contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 查找最大轮廓
max_contour = max(contours, key=cv2.contourArea)

# 绘制最大轮廓
cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 3)

# 显示结果
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
