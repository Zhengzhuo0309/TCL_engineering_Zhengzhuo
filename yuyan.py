import cv2

# 读取原始图像
img = cv2.imread('1.jpg')

# 相机矩阵
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]]

# 畸变系数
dist = [k1, k2, p1, p2]

# 消除畸变
dst = cv2.undistort(img, np.array(K), np.array(dist))

# 显示消除畸变后的图像
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
