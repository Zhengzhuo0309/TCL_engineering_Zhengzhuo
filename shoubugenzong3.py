import cv2
import numpy as np

# 将手部坐标转换为OpenCV的Point类型
hand_point = (100, 200)
hand_point_cv = cv2.Point(hand_point[0], hand_point[1])

# 获取显示屏边框的坐标
contours = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
contours_cv = [cv2.Point(x, y) for x, y in contours]

# 判断手部坐标是否在显示屏边框内部
inside = cv2.pointPolygonTest(contours_cv, hand_point_cv, False)

if inside >= 0:
    print('手部坐标在显示屏边框内部')
else:
    print('手部坐标在显示屏边框外部')
