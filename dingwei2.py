import cv2

# 读取图像并进行预处理
img = cv2.imread("1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150)

# 进行轮廓检测
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 进行形状匹配
for contour in contours:
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    # 进行轮廓近似
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    # 如果轮廓近似为矩形，则认为是显示屏边框
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.7 <= aspect_ratio <= 1.3:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 显示结果
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
