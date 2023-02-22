import cv2
import numpy as np

# 读取图像
img = cv2.imread('screen.jpg')

# 定义显示屏边框
screen_contour = np.array([[50, 50], [250, 50], [250, 450], [50, 450]])

# 定义鼠标回调函数，模拟手部运动点
mouse_position = None
def mouse_callback(event, x, y, flags, param):
    global mouse_position
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_position = (x, y)

# 创建窗口并绑定鼠标回调函数
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

# 创建Lucas-Kanade光流法对象
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化跟踪点
track_point = None

# 开始循环
while True:
    # 读取图像并进行灰度化
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测手部运动点
    if mouse_position is not None:
        track_point = np.array(mouse_position).astype(np.float32)
        mouse_position = None

    # 进行光流法跟踪
    if track_point is not None:
        new_track_point, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, track_point.reshape(-1, 1, 2), None, **lk_params)
        track_point = new_track_point.ravel()

        # 绘制跟踪点
        cv2.circle(frame, tuple(track_point.astype(int)), 5, (0, 255, 0), -1)

        # 计算手部运动点到显示屏边框的距离
        if screen_contour is not None:
            dist = cv2.pointPolygonTest(screen_contour, tuple(track_point.astype(int)), True
