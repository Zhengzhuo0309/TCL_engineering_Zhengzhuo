import cv2

# 读取原始图像
img = cv2.imread('fish_eye_image.jpg')

# 相机矩阵
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]]

# 畸变系数
dist = [k1, k2, k3, k4]

# 使用鱼眼畸变校正函数进行校正
mapx, mapy = cv2.fisheye.initUndistortRectifyMap(np.array(K), np.array(dist), np.eye(3), K, img.shape[:2], cv2.CV_16SC2)
dst = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# 显示消除畸变后的图像
cv2.imshow('Undistorted Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
