import cv2
import numpy as np

# 读入图片
img = cv2.imread("screen.png")

# 定义边框坐标
contour = np.array([(108, 47), (216, 47), (216, 167), (108, 167)])

# 缩小边框10%作为手部运动的有效区域
border = int(min(img.shape[:2]) * 0.1)
hand_contour = contour + border
hand_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.drawContours(hand_mask, [hand_contour], 0, 255, -1)

# 获取手部运动坐标，并将其转换为与边框坐标系相同的坐标系，以便进行匹配
hand_pos = (100, 100)
hand_pos = np.array([hand_pos[0] - hand_contour[0][0], hand_pos[1] - hand_contour[0][1]])

# 遍历边框坐标，计算每个边框点与手部运动坐标的距离
distances = []
for point in contour:
    point_dist = np.linalg.norm(hand_pos - (point - hand_contour[0]))
    distances.append(point_dist)

# 将距离小于设定阈值的边框点加入匹配点列表，并记录其距离
threshold = 20
match_points = []
for i in range(len(contour)):
    if distances[i] < threshold:
        match_points.append(contour[i])

# 对匹配点列表按照距离进行排序，取前n个点作为匹配结果
match_points = sorted(match_points, key=lambda x: distances[contour.tolist().index(x)])
n = 3
match_points = match_points[:n]

# 绘制匹配点
for point in match_points:
    cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)

# 显示结果
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
