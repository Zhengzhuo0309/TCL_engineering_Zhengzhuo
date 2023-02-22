import cv2
import numpy as np

# 假设边框坐标为frame_coords，手部运动坐标为hand_coords

# 设定边框缩小比例
frame_scale = 0.2

# 获取边框中心点
frame_center = np.mean(frame_coords, axis=0)

# 计算边框缩小后的边界坐标
frame_min = np.min(frame_coords, axis=0)
frame_max = np.max(frame_coords, axis=0)
frame_size = frame_max - frame_min
frame_min_new = frame_center - frame_size / 2 * (1 - frame_scale)
frame_max_new = frame_center + frame_size / 2 * (1 - frame_scale)

# 获取手部运动坐标中心点
hand_center = np.mean(hand_coords, axis=0)

# 计算手部运动坐标相对于边框的偏移
offset = hand_center - frame_center

# 将边框坐标和手部运动坐标都平移到以边框中心点为原点的坐标系中
frame_coords_new = frame_coords - frame_center
hand_coords_new = hand_coords - frame_center

# 将手部运动坐标缩放到边框坐标系中
hand_coords_new *= (frame_max_new - frame_min_new) / (frame_max - frame_min)
hand_coords_new += frame_min_new

# 计算每个边框点与手部运动坐标的距离，并将距离小于设定阈值的边框点加入匹配点列表
match_points = []
threshold = 30
for i in range(len(frame_coords)):
    dist = np.linalg.norm(frame_coords_new[i] - hand_coords_new)
    if dist < threshold:
        match_points.append((frame_coords[i], dist))

# 对匹配点列表按照距离进行排序，取前n个点作为匹配结果
n = 3
match_points.sort(key=lambda x: x[1])
match_results = [p[0] for p in match_points[:n]]

# 根据匹配结果进行相应的处理，例如绘制匹配点或者执行相应的操作
img = cv2.imread("image.jpg")
for point in match_results:
    cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
cv2.imshow("Match Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
