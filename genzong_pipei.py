import cv2
import numpy as np

# 获取边框坐标，这里假设已经获取了边框坐标并保存在coords变量中
coords = np.array([[10, 10], [10, 100], [100, 100], [100, 10]])

# 获取手部运动坐标，这里假设已经获取了手部运动坐标并保存在hand_coords变量中
hand_coords = np.array([50, 50])

# 将手部运动坐标转换为与边框坐标系相同的坐标系
hand_coords_transformed = hand_coords - coords[0]

# 遍历边框坐标，计算每个边框点与手部运动坐标的距离
match_points = []
for coord in coords:
    dist = np.linalg.norm(coord - hand_coords_transformed)
    if dist < threshold:
        match_points.append((coord, dist))

# 对匹配点列表按照距离进行排序，取前n个点作为匹配结果
match_points_sorted = sorted(match_points, key=lambda x: x[1])
n = min(len(match_points_sorted), n)
match_result = [match_points_sorted[i][0] for i in range(n)]

# 可以根据需要进行相应的处理
