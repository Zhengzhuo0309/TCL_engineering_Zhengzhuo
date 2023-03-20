import cv2
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
# # 读取图像
# img = cv2.imread('data/frame_135.png')
# # img = cv2.imread('img_large.jpeg')
# img = cv2.resize(img,(1920,1080))
# # 转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray_copy = copy.deepcopy(gray)
def egde_detect_biology(gray):
    # 进行阈值分割
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # 进行形态学操作，去除噪声和小区域
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 进行连通区域分析
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)
    area_label_list = list(np.where((stats[:, 4] < 1000000) & (stats[:, 4] > 100000))[0])
    # #找到最大连通区域
    # max_area = 0
    # max_label = 0
    # for i in range(0, n):
    #     if stats[i, cv2.CC_STAT_AREA] > max_area:
    #         max_area = stats[i, cv2.CC_STAT_AREA]
    #         max_label = i
    # stats[np.argsort(-stats[:, 4])][1:3, :]
    # # 提取最大连通区域
    # mask = np.zeros_like(gray)
    # mask[labels == max_label] = 255

    # 提取最大连通区域
    mask = np.zeros_like(gray)
    for area_label in area_label_list:
        mask[labels == area_label] = 255

    # 进行形态学操作，填充区域
    kernel_5x5 = np.ones((5, 5), np.uint8)
    kernel_3x3 = np.ones((3, 3), np.uint8)
    mask_er = cv2.dilate(mask,kernel_5x5,iterations=50)
    closed_mask = cv2.morphologyEx(mask_er, cv2.MORPH_CLOSE, kernel_5x5)

    # # 进行轮廓检测
    # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask,closed_mask,mask_er
# 筛选出特定大小和形状的轮廓
# mask,closed_mask,mask_er= egde_detect_biology(gray)

# mask最大区域
def find_mask_max(closed_mask,gray):
    gray_copy = copy.deepcopy(gray)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(255-closed_mask,connectivity=4)
    max_label = np.where(stats[:,4]==stats[1:,4].max())[0][0]
    max_label = max_label.astype(np.int32)
    gray_mask_bool = ~(labels==max_label)
    gray_mask = np.zeros_like(gray)
    gray_mask[gray_mask_bool]=255
    gray_copy[gray_mask_bool]=255
    return gray_copy,gray_mask,gray_mask_bool
# gray,gray_mask,gray_mask_bool = find_mask_max(closed_mask,gray)

# 过滤矩形外区域
def rect_mask_filter(gray_mask,gray):
    gray_mask_copy = copy.deepcopy(gray_mask)
    gray_copy = copy.deepcopy(gray)
    mask_num = np.sum(gray_mask==0,1)
    di = np.diff(mask_num)
    di = np.append(np.zeros(1,int),di)
    y_up = np.where(di==di.max())[0][0]
    y_down = np.where(di==di.min())[0][0]
    gray_copy[:y_up+1,:]=255
    gray_copy[y_down:,:]=255
    gray_mask_copy[:y_up+1,:]=255
    gray_mask_copy[y_down:,:]=255
    # plt.figure(1)
    # plt.subplot(1,2,1)
    # plt.plot(list(range(len(mask_num))),mask_num)
    # plt.subplot(1,2,2)
    # plt.plot(list(range(len(di))),di)
    # plt.show()
    return gray_copy,gray_mask_copy
# gray,gray_mask = rect_mask_filter(gray_mask,gray)
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.plot(list(range(len(mask_num))),mask_num)
# plt.subplot(1,2,2)
# plt.plot(list(range(len(di))),di)
# plt.show()
#
# # 膨胀回来
# kernel_5x5 = np.ones((5, 5), np.uint8)
# gray_mask_ode = cv2.erode(gray_mask,kernel_5x5,iterations=6)

# 边缘检测
def findcountours(gray_mask_ode):
    contours, hierarchy = cv2.findContours(255-gray_mask_ode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    screen_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        if aspect_ratio > 1 and aspect_ratio < 5:
            screen_contours.append(contour)
    return tuple(screen_contours)
# screen_contours = findcountours(gray_mask_ode)


# 前后边框检测——提高鲁棒性
def video_countours(img,screen_contours):
    pass

# 内接矩形
def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect

# 转换轨迹格式及计算长度
def contours_to_tuple_and_len(screen_contours_best_left_xy):  # 格式 np.array([[x1,y1]，[x2,y2]...])
    screen_contours_best_left = tuple(np.array([list(np.expand_dims(screen_contours_best_left_xy, 1))]))
    contour_left_length = cv2.arcLength(screen_contours_best_left[0], False)
    return screen_contours_best_left,contour_left_length
# 轨迹按区域分段
def dividing_contours(scb,box_buffer,box_in_buffer):  # 格式 np.array([[x1,y1]，[x2,y2]...])
    x_inbuffer_left, x_inbuffer_right, y_inbuffer_up, y_inbuffer_down = box_in_buffer[0][0], box_in_buffer[1][0], \
                                                                        box_in_buffer[1][1], box_in_buffer[2][1]
    xbuffer_left, xbuffer_right, ybuffer_up, ybuffer_down = box_buffer[0][0], box_buffer[1][0], \
                                                            box_buffer[1][1], box_buffer[2][1]
    screen_contours_best_left_xy = scb[(xbuffer_left <= scb[:, 0]) & (scb[:, 0] <= x_inbuffer_left)]
    screen_contours_best_right_xy = scb[(x_inbuffer_right <= scb[:, 0]) & (scb[:, 0] <= xbuffer_right)]
    screen_contours_best_up_xy = scb[(ybuffer_up <= scb[:, 1]) & (scb[:, 1] <= y_inbuffer_up)]

    screen_contours_best_left,contour_left_length = contours_to_tuple_and_len(screen_contours_best_left_xy)
    screen_contours_best_right, contour_right_length = contours_to_tuple_and_len(screen_contours_best_right_xy)
    screen_contours_best_up, contour_up_length = contours_to_tuple_and_len(screen_contours_best_up_xy)

    return (screen_contours_best_left_xy, screen_contours_best_right_xy, screen_contours_best_up_xy), \
           (screen_contours_best_left, screen_contours_best_right, screen_contours_best_up), \
           (contour_left_length, contour_right_length, contour_up_length)

# 轨迹点是否在有效区域内
def is_in_area(point,box_buffer,box_in_buffer): # 格式 np.array([x,y]):
    x_inbuffer_left, x_inbuffer_right, y_inbuffer_up, y_inbuffer_down = box_in_buffer[0][0], box_in_buffer[1][0], \
                                                                        box_in_buffer[1][1], box_in_buffer[2][1]
    xbuffer_left, xbuffer_right, ybuffer_up, ybuffer_down = box_buffer[0][0], box_buffer[1][0], \
                                                            box_buffer[1][1], box_buffer[2][1]
    is_in_left = (xbuffer_left <= point[0]) & (point[0] <= x_inbuffer_left)
    is_in_right = (x_inbuffer_right <= point[0]) & (point[0] <= xbuffer_right)
    is_in_up = (ybuffer_up <= point[1]) & (point[1] <= y_inbuffer_up)
    in_area = is_in_left | is_in_right | is_in_up
    return in_area
# 轨迹直线拟合
def fit_contours_vertical(screen_contours_best_left_xy):  # 输入一边纵向轨迹，拟合直线; 格式 np.array([[x1,y1]，[x2,y2]...])
    block_left_y_max = screen_contours_best_left_xy[:, 1].max()
    block_left_y_min = screen_contours_best_left_xy[:, 1].min()
    # rows, cols = img_copy.shape[:2]
    [vx, vy, x_point, y_point] = cv2.fitLine(screen_contours_best_left_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    k_fit = vy / vx
    b_fit = y_point - k_fit * x_point
    # lefty = int((-x_point * vy / vx) + y_point)
    # righty = int(((cols - x_point) * vy / vx) + y_point)
    if np.count_nonzero(k_fit)!=0:
        left_x_max = int((block_left_y_max - b_fit) / k_fit)
        left_x_min = int((block_left_y_min - b_fit) / k_fit)
    else:                                      #k=0
        left_x_max = screen_contours_best_left_xy[:, 0].max()
        left_x_min = screen_contours_best_left_xy[:, 0].min()
    track_length = np.linalg.norm(np.array([left_x_min, block_left_y_min]) - np.array([left_x_max, block_left_y_max]))

    return k_fit, b_fit, track_length, (left_x_min, block_left_y_min), (left_x_max, block_left_y_max)

def fit_contours_horizontal(screen_contours_best_up_xy):  # 输入一边横向轨迹，拟合直线; 格式 np.array([[x1,y1]，[x2,y2]...])
    block_up_x_max = screen_contours_best_up_xy[:, 0].max()
    block_up_x_min = screen_contours_best_up_xy[:, 0].min()
    # rows, cols = img_copy.shape[:2]
    [vx, vy, x_point, y_point] = cv2.fitLine(screen_contours_best_up_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    k_fit = vy / vx
    b_fit = y_point - k_fit * x_point
    # lefty = int((-x_point * vy / vx) + y_point)
    # righty = int(((cols - x_point) * vy / vx) + y_point)
    up_y_max = int(k_fit*block_up_x_max+b_fit)
    up_y_min = int(k_fit*block_up_x_min+b_fit)
    track_length = np.linalg.norm(np.array([block_up_x_min,up_y_min]) - np.array([block_up_x_max,up_y_max]))
    return k_fit, b_fit, track_length, (block_up_x_min,up_y_min), (block_up_x_max,up_y_max)

def projection(point,k,b):    #求投影后坐标
    x = (k*(point[1]-b)+point[0])/(k**2+1)
    y = k*x+b
    return int(x),int(y)

def projection_length(track_minpoint,track_maxpoint, k_fit, b_fit):  #轨迹投影后坐标与长度
    track_left_x_min, track_left_y_min = projection(track_minpoint, k_fit, b_fit)
    track_left_x_max, track_left_y_max = projection(track_maxpoint, k_fit, b_fit)
    track_length_pro = np.linalg.norm(np.array([track_left_x_min, track_left_y_min]) - np.array([track_left_x_max, track_left_y_max]))
    track_minpoint_pro = (track_left_x_min, track_left_y_min)
    track_maxpoint_pro = (track_left_x_max, track_left_y_max)
    return track_minpoint_pro,track_maxpoint_pro,track_length_pro

# 决策字典更新
def update_match_dict(match_dict, position, direction, end, length):
    match_dict[position][direction]['end_point'] = end
    match_dict[position][direction]['need_length'] = length

# 出界时找轨迹变化方向,两个方向次数一样时怎么处理
def find_direction_whenout(fit_change_num_x_minus,fit_change_num_x_plus,fit_change_num_y_minus,fit_change_num_y_plus):
    direction_dict = {'x_minus':fit_change_num_x_minus,'x_plus':fit_change_num_x_plus,
                      'y_minus':fit_change_num_y_minus,'y_plus':fit_change_num_y_plus}
    max_num = max(fit_change_num_x_minus,fit_change_num_x_plus,fit_change_num_y_minus,fit_change_num_y_plus)
    direction_max = [k for k, v in direction_dict.items() if v == max_num]
    if max_num==0:
        return ''
    if direction_max:
        if len(direction_max)==1:
            return direction_max[0]
        else:                         #多个方向时取哪个方向，这里还没改好的！！！！！！
            return direction_max[0]
    else:
        return ''
# 界内找轨迹变化方向
def find_direction_whenin(fit_change_num_x_minus, fit_change_num_x_plus,fit_change_num_y_minus,fit_change_num_y_plus,in_time_count_th):
    if fit_change_num_x_minus>=in_time_count_th:
        return 'x_minus'
    elif fit_change_num_x_plus>=in_time_count_th:
        return 'x_plus'
    elif fit_change_num_y_minus>=in_time_count_th:
        return 'y_minus'
    elif fit_change_num_y_plus>=in_time_count_th:
        return 'y_plus'
    else:
        return ''
#确定轨迹点位置
def find_position(point,box_buffer,box_in_buffer):
    x_inbuffer_left, x_inbuffer_right, y_inbuffer_up, y_inbuffer_down = box_in_buffer[0][0], box_in_buffer[1][0], \
                                                                        box_in_buffer[1][1], box_in_buffer[2][1]
    xbuffer_left, xbuffer_right, ybuffer_up, ybuffer_down = box_buffer[0][0], box_buffer[1][0], \
                                                            box_buffer[1][1], box_buffer[2][1]
    is_in_left = (xbuffer_left <= point[0]) & (point[0] <= x_inbuffer_left)
    is_in_right = (x_inbuffer_right <= point[0]) & (point[0] <= xbuffer_right)
    # 不能用两个都不在的，因为边框可能实时变化
    if is_in_left:
        return 'left'
    else:
        return 'right'


def reset_direction_count():        # 重置方向计数
    fit_change_num_x_plus = 0       # 拟合点x正方向变化计数
    fit_change_num_y_plus = 0       # 拟合点y正方向变化计数
    fit_change_num_x_minus = 0      # 拟合点x负方向变化计数
    fit_change_num_y_minus = 0      # 拟合点y负方向变化计数
    return fit_change_num_x_plus,fit_change_num_y_plus,fit_change_num_x_minus,fit_change_num_y_minus

def reset():     #重置所有计数器及容器 —— 拟合一次后调用
    track_cur = []            # 当前用于拟合的轨迹
    track_cur_img = []        # 当前用于拟合的轨迹对应图片
    move_cur = []             # 被移除的轨迹点——下一段轨迹的开头
    update_flag = False       # 长度匹配开关，分出一个轨迹段之后打开，匹配一次后关闭
    position = ''                   # 当前位置
    direction = ''                  # 当前轨迹方向
    change_direction = ''           # 轨迹点变化方向

    in_time_count = 0               # 处于该方向的次数
    out_time_count = 0              # 该方向确定之后，不在该方向的次数
    out_count = 0                   # 跑出有效区域个数
    is_out_count_over = False       # 代表是区域外计数足够了
    out_time_count_flag = False     # 其他方向计数开关

    return track_cur,track_cur_img,move_cur,update_flag,\
           position,direction,change_direction,in_time_count,out_time_count,\
           out_count,is_out_count_over,out_time_count_flag
class Config(object): #配置文件
    def __init__(self):
        pass
    def reset(self):     #重置所有计数器及容器 —— 拟合一次后调用
        pass
# # 创建新的图像并绘制分割轮廓
# seg_img = np.zeros_like(img)
# img_copy = copy.deepcopy(img)
# cv2.drawContours(img_copy, screen_contours, -1, (0, 0, 255), 2)
#
# gray_copy[closed_mask==255]=255
#
# # 显示结果
# kernel_5x5 = np.ones((5, 5), np.uint8)
# kernel_3x3 = np.ones((3, 3), np.uint8)
# mask_test = cv2.dilate(gray_mask, kernel_5x5, iterations=6)
# # opening = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel_5x5, iterations=10)
# cv2.namedWindow("Segmented Image", 0)
# cv2.resizeWindow("Segmented Image",1920, 1080)  # 设置窗口大小
# cv2.imshow('Segmented Image',img_copy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    import re
    def pngfilter(f):
        if f[-4:] in ['.jpg', '.png', '.bmp']:
            return True
        else:
            return False
    data_root = os.getcwd() + '/data_310'
    save_root = os.getcwd() + '/vis_img_310'
    img_file_list_with_sub = os.listdir(os.getcwd() + '/data_310')
    img_file_list_rand = list(filter(pngfilter, img_file_list_with_sub))
    img_file_list = sorted(img_file_list_rand, key=lambda x: int(re.findall(r"\d+", x)[0]))
    track = pd.read_csv("/hhd/Zhengzhuo/result_yifan.csv")
    track_xy = track[['x', 'y']].values
    track_point_per = track_xy[0]   # 上一时刻的轨迹点
    track_cur = []            # 当前用于拟合的轨迹
    track_cur_img = []        # 当前用于拟合的轨迹对应图片
    move_cur = []             # 被移除的轨迹点——下一段轨迹的开头
    track_all = []            # 所有轨迹
    track_img_all = []      # 所有轨迹对应的图片
    track_direction_all = []      # 所有轨迹对应的方向
    update_flag = False       # 长度匹配开关，分出一个轨迹段之后打开，匹配一次后关闭

    position = ''                   # 当前位置
    direction = ''                  # 当前轨迹方向
    change_direction = ''           # 轨迹点变化方向
    fit_change_num_x_plus = 0       # 拟合点x正方向变化计数
    fit_change_num_y_plus = 0       # 拟合点y正方向变化计数
    fit_change_num_x_minus = 0      # 拟合点x负方向变化计数
    fit_change_num_y_minus = 0      # 拟合点y负方向变化计数
    fit_change_th = 20              # 拟合变化计数阈值

    in_time_count = 0               # 处于该方向的次数
    in_time_count_th = 5           # 处于该方向的次数阈值
    out_time_count = 0              # 该方向确定之后，不在该方向的次数
    out_time_count_th = 10          # 不在该方向上次数阈值
    out_count = 0                   # 跑出有效区域个数
    out_count_th = 10               # 区域外次数阈值
    is_out_count_over = False       # 代表是区域外计数足够了
    out_time_count_flag = False     # 其他方向计数开关

    match_dict = {'left':           #决策匹配字典
                    {
                        'y_plus':{'func':fit_contours_vertical,'end_point':None,'need_length':None},
                        'y_minus':{'func':fit_contours_vertical,'end_point':None,'need_length':None}
                    },
                'right':
                    {
                        'y_plus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None},
                        'y_minus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None}
                    },
                'up':
                    {
                        'x_plus': {'func': fit_contours_horizontal, 'end_point': None, 'need_length': None},
                        'x_minus': {'func': fit_contours_horizontal, 'end_point': None, 'need_length': None}
                    },
                  }
    match_dict_start = {}      #边框匹配字典
    done_dict = {'left':0,'right':0,'up':0}

    wait_time = 0.5
    i = 1
    break_for = False
    screen_contours_old = []
    screen_contours_new = []
    rect_all = []
    box_all = []
    box_all_new = []
    area_per_list = []
    area_per_min_list = []
    area_per_min = 0
    screen_contours_best = None
    box_best = None
    count_keep = 0
    count_keep_max = 3 #保持次数
    dist = 100000
    dist_threshold = 3 #距离阈值
    buffer = 10 #扩展像素点
    buffer_right = 40 #右外边框额外扩展像素点
    area_per_change_threshold = 0.005   #面积比变化阈值
    for image_path,track_point in zip(img_file_list,track_xy):
        if break_for:
            image_path  = 'frame_355.png'
            # image_path = 'frame_3525.png'buffer
            # image_path = 'frame_4800.png'
            # image_path = 'frame_120.png'
            # image_path = 'frame_890.png'
        image_path_root = os.path.join(data_root, image_path)

        # image_path = "798.jpg"
        # image_path_root = "798.jpg"

        img = cv2.imread(image_path_root)
        # img = cv2.imread('img_large.jpeg')
        img = cv2.resize(img, (1920, 1080))

        seg_img = np.zeros_like(img)
        img_copy = copy.deepcopy(img)
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_copy = copy.deepcopy(gray)
        mask, closed_mask, mask_er = egde_detect_biology(gray_copy)
        gray_max, gray_mask_max, gray_mask_max_bool = find_mask_max(closed_mask, gray_copy)
        gray, gray_mask = rect_mask_filter(gray_mask_max, gray_max)

        # 膨胀回来
        kernel_5x5 = np.ones((5, 5), np.uint8)
        gray_mask_ode = cv2.erode(gray_mask_max, kernel_5x5, iterations=50)
        screen_contours = findcountours(gray_mask_ode)
        screen_contours_old.append(screen_contours)

        #外接矩形
        rect = cv2.minAreaRect(screen_contours[0])
        rect_all.append(rect)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        box_all.append(box)   #没有经过排序

        # cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)

        area = cv2.contourArea(box)
        area_contour = cv2.contourArea(screen_contours[0])
        area_per = area_contour/area#面积比例


        #边框迭代优化
        # -------面积比变大暂存---------
        if (area_per_min < area_per):
            screen_contours_best_tmp = screen_contours
            area_per_min_tmp = area_per
            box_tmp = box
        # -------面积比变大暂存---------
        #1. area_per_min > area_per_cur  #留下 area_per_min = area_per_cur
        # 2.角点变化  #新SN area_per_min = area_per_cur
        if len(box_all)>count_keep_max:

            box_past = box_all[-2]   #上一次识别框外接矩形

            #判断识别框是否长时间不变
            M_past = cv2.moments(box_past)
            cx_past = int(M_past['m10'] / M_past['m00'])
            cy_past = int(M_past['m01'] / M_past['m00'])
            point_past = np.array([cx_past, cy_past])
            M = cv2.moments(box)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            point = np.array([cx, cy])
            dist = np.linalg.norm(point_past-point)      #重心变化
            #-------重心保持---------
            if (dist < dist_threshold):
                count_keep += 1
            else:
                count_keep = 0
            # -------重心保持---------

            # -------重心保持暂存---------
            if (count_keep >= count_keep_max):
                screen_contours_best_tmp = screen_contours
                box_tmp = box
                area_per_min_tmp = area_per
            # -------重心保持---------

            # -------处理初始帧-----
        else:
            screen_contours_best = screen_contours_best_tmp
            area_per_min = area_per_min_tmp
            box_best = box_tmp
            # -------处理初始三帧-----

        # -------重心保持覆盖---------   条件：面积比变小程度不大，重心变化不大，并且保持了一段时间
        if (dist < dist_threshold) and (count_keep >= count_keep_max) and ((area_per_min - area_per) < area_per_change_threshold):

            screen_contours_best = screen_contours_best_tmp
            box_best = box_tmp
            area_per_min = area_per_min_tmp
            count_keep = 0
        # -------重心保持覆盖---------

        area_per_list.append(area_per)
        area_per_min_list.append(area_per_min)

        #box排序，便于后面显示可用区域
        box_best_xs = [i[0] for i in box_best]
        box_best_ys = [i[1] for i in box_best]
        box_best_xs.sort()
        box_best_ys.sort()
        box_best = np.array([[box_best_xs[1],box_best_ys[1]],[box_best_xs[2],box_best_ys[1]],
                           [box_best_xs[2],box_best_ys[2]],[box_best_xs[1],box_best_ys[2]]])

        #最比例最大的（找到最好的框）---->  框长时间不变，开启角点检测  ----> 角点变化，当做新的SN检测，但是保留轨迹
        screen_contours_new.append(screen_contours_best)
        box_all_new.append(box_best)  #经过排序后

        ##是否需要新的外接矩形
        # 创建新的图像并绘制分割轮廓
        img_test = np.zeros_like(img_copy)
        cv2.drawContours(img_test, screen_contours_best, -1, (0, 0, 255), 2)
        cv2.drawContours(img_copy, screen_contours_best, -1, (0, 0, 255), 2)
        # cv2.drawContours(img_copy, [box_best], -1, (255, 0, 0), 2)
        # gray_copy[closed_mask == 255] = 255

        #内接矩形
        rect_in = order_points(screen_contours_best[0].reshape(screen_contours_best[0].shape[0], 2))
        rect_in = np.int64(rect_in)
        xs_in = [i[0] for i in rect_in]
        ys_in = [i[1] for i in rect_in]
        xs_in.sort()
        ys_in.sort()
        # 内接矩形的坐标为
        # print(xs_in[1], xs_in[2], ys_in[1], ys_in[2])
        box_in = np.array([[xs_in[1],ys_in[1]],[xs_in[2],ys_in[1]],
                           [xs_in[2],ys_in[2]],[xs_in[1],ys_in[2]]])
        box_in_buffer = np.array([[xs_in[1]+buffer,ys_in[1]+buffer],[xs_in[2]-buffer,ys_in[1]+buffer],
                                  [xs_in[2]-buffer,ys_in[2]-buffer],[xs_in[1]+buffer,ys_in[2]-buffer]])
        # cv2.drawContours(img_copy, [box_in], -1, (0, 255, 255), 2)
        x_inbuffer_left, x_inbuffer_right, y_inbuffer_up, y_inbuffer_down = box_in_buffer[0][0], box_in_buffer[1][0], \
                                                                   box_in_buffer[1][1], box_in_buffer[2][1]
        #有效区域
        box_buffer = np.array([[box_best[0][0] - buffer, box_best[0][1] - buffer], [box_best[1][0] + buffer+ buffer_right, box_best[1][1] - buffer],
                               [box_best[2][0] + buffer + buffer_right, box_best[2][1] + buffer], [box_best[3][0] - buffer, box_best[3][1] + buffer]])
        xbuffer_left, xbuffer_right, ybuffer_up, ybuffer_down = box_buffer[0][0], box_buffer[1][0], \
                                                                   box_buffer[1][1], box_buffer[2][1]

        rectangle_out_mask = np.zeros(img.shape[0:2], dtype="uint8")
        cv2.rectangle(rectangle_out_mask, tuple(box_buffer[0]), tuple(box_buffer[2]), 255, -1)
        rectangle_in_mask = np.zeros(img.shape[0:2], dtype="uint8")
        cv2.rectangle(rectangle_in_mask, tuple(box_in_buffer[0]), tuple(box_in_buffer[2]), 255, -1)
        buffer_area = rectangle_out_mask-rectangle_in_mask
        buffer_area_bool = (buffer_area==255)
        area_zeros1 = 255-np.zeros_like(buffer_area)
        area_zeros1[buffer_area_bool] = 0
        area_zeros2 = 255-np.zeros_like(buffer_area)
        area_zeros2[buffer_area_bool] = 0
        area_zeros3 = 255 - np.zeros_like(buffer_area)
        buffer_area_img = np.dstack((area_zeros1, area_zeros3, area_zeros2))
        # buffer_area_img = np.repeat(np.expand_dims(buffer_area, axis=2),3,2)
        # img_copy[buffer_area_bool] = 255

        #轨迹模拟
        contours_rand = np.random.randint(-int(buffer/2), int(buffer/2), screen_contours_best[0].shape, np.int32)
        screen_contours_best_rand = tuple([screen_contours_best[0]+contours_rand])
        # cv2.drawContours(img_copy, screen_contours_best_rand, -1, (255, 255, 255), 2)
        # cv2.drawContours(img_test, screen_contours_best_rand, -1, (255, 255, 255), 2)
        #分割曲线
        # ---------------------------------------------------------------------------------------------------     边框
        scb = screen_contours_best[0].squeeze()
        screen_contours_best_part_xy,screen_contours_best_part,screen_contours_best_part_length = dividing_contours(scb,box_buffer,box_in_buffer)
        (screen_contours_best_left_xy, screen_contours_best_right_xy, screen_contours_best_up_xy) = screen_contours_best_part_xy
        (screen_contours_best_left, screen_contours_best_right, screen_contours_best_up) = screen_contours_best_part
        (contour_left_length, contour_right_length, contour_up_length) = screen_contours_best_part_length
        # print("边框周长：", contour_left_length)
        #拟合曲线
        k_fit, b_fit, track_length,minpoint,maxpoint = fit_contours_vertical(screen_contours_best_left_xy)
        (left_x_min, left_y_min) = minpoint
        (left_x_max, left_y_max) = maxpoint
        # print("左边框长度：", track_length)
        # cv2.line(img_copy, minpoint, maxpoint, (255, 0, 0), 2)

        k_fit_r, b_fit_r, track_length_r, minpoint_r, maxpoint_r = fit_contours_vertical(screen_contours_best_right_xy)
        # print("右边框长度：", track_length_r)
        (right_x_min, right_y_min) = minpoint_r
        (right_x_max, right_y_max) = maxpoint_r
        # cv2.line(img_copy, minpoint_r, maxpoint_r, (255, 0, 0), 2)

        k_fit_u, b_fit_u, track_length_u, minpoint_u, maxpoint_u = fit_contours_horizontal(screen_contours_best_up_xy)
        (up_x_min, up_y_min) = minpoint_u
        (up_x_max, up_y_max) = maxpoint_u
        # print("上边框长度：", track_length_u)
        # cv2.line(img_copy, minpoint_u, maxpoint_u, (255, 0, 0), 2)

        # 初始化match_dict
        update_match_dict(match_dict,'left','y_plus',left_y_min,track_length)
        update_match_dict(match_dict, 'left', 'y_minus', left_y_max, track_length)
        update_match_dict(match_dict,'right','y_plus',right_y_min,track_length_r)
        update_match_dict(match_dict, 'right', 'y_minus', right_y_max, track_length_r)
        update_match_dict(match_dict,'up','x_plus',up_x_min,track_length_u)
        update_match_dict(match_dict, 'up', 'x_minus', up_x_max, track_length_u)
        if not match_dict_start:
            match_dict_start = copy.deepcopy(match_dict)       #原始边框需要匹配的信息

        # ---------------------------------------------------------------------------------------------------
        diff_x = track_point[0] - track_point_per[0]
        diff_y = track_point[1] - track_point_per[1]
        # 计数
        if is_in_area(track_point,box_buffer,box_in_buffer):          #有效区域内
            if diff_x > fit_change_th:       #往x正方向变化
                fit_change_num_x_plus+=1
                change_direction = 'x_plus'
            if diff_x < -fit_change_th:      #往x负方向变化
                fit_change_num_x_minus+=1
                change_direction = 'x_minus'
            if diff_y > fit_change_th:       #往y正方向变化
                fit_change_num_y_plus+=1
                change_direction = 'y_plus'
            if diff_y < -fit_change_th:      #往y负方向变化
                fit_change_num_y_minus+=1
                change_direction = 'y_minus'
            track_cur.append(track_point)
            track_cur_img.append(image_path)
            track_point_per = track_point
        else:                                #有效区域外
            out_count+=1
            print(track_point)
        # 分段
        if out_count>=out_count_th:
            direction = find_direction_whenout(fit_change_num_x_minus, fit_change_num_x_plus,
                                               fit_change_num_y_minus,fit_change_num_y_plus)
            print('direction:', direction,end=' ')
            is_out_count_over = True     #标记轨迹列表要全部清空

            #有方向就拟合
            if direction and track_cur:
                update_flag = True
                # 确定位置
                if 'y' in direction:
                    position = find_position(track_cur[-1],box_buffer,box_in_buffer) #用有效轨迹的最后一个点判断位置，防止最后一个轨迹点是无效的
                else:
                    position = 'up'
                func = match_dict[position][direction]['func']
                end_point = match_dict[position][direction]['end_point']
                need_length = match_dict[position][direction]['need_length']
                k_track_fit, b_track_fit, track_true_length, track_minpoint, track_maxpoint = func(np.array(track_cur))
                (track_x_min, track_y_min) = track_minpoint
                (track_x_max, track_y_max) = track_maxpoint
                track_minpoint_pro, track_maxpoint_pro, track_length_pro = projection_length(track_minpoint, track_maxpoint,
                                                                                             k_fit, b_fit)
                print("真实轨迹长度：", track_true_length, "投影长度：", track_length_pro)
                cv2.line(img_copy, track_minpoint, track_maxpoint, (255, 255, 0), 2)

            # 更新
            if update_flag:
                if direction == 'x_plus':
                    match_dict[position][direction]['end_point'] = track_x_max
                elif direction == 'x_minus':
                    match_dict[position][direction]['end_point'] = track_x_min
                elif direction == 'y_plus':
                    match_dict[position][direction]['end_point'] = track_y_max
                elif direction == 'y_minus':
                    match_dict[position][direction]['end_point'] = track_y_min
                match_dict[position][direction]['need_length'] = match_dict[position][direction]['need_length'] - track_length_pro  #同个方向的都拼一块了，没有判断起点终点！！！！！
                if match_dict[position][direction]['need_length'] <= 0:
                    match_dict[position][direction]['need_length'] = match_dict_start[position][direction]['need_length']
                    match_dict[position][direction]['end_point'] = match_dict_start[position][direction]['end_point']
                    done_dict[position] +=1    #画好一条边
                    print('Done ' + position)
                    print(done_dict)
            # 清除
            track_all.append(track_cur)
            track_img_all.append(track_cur_img)
            track_direction_all.append(direction)
            track_cur,track_cur_img, move_cur,  update_flag, \
            position, direction, change_direction, in_time_count, out_time_count, \
            out_count, is_out_count_over, out_time_count_flag = reset()
        else:
            direction = find_direction_whenin(fit_change_num_x_minus, fit_change_num_x_plus,
                                               fit_change_num_y_minus,fit_change_num_y_plus,in_time_count_th)
            print('direction:', direction,end=' ')
            if direction and is_in_area(track_point,box_buffer,box_in_buffer):
                out_time_count_flag = True
                fit_change_num_x_plus, fit_change_num_y_plus, fit_change_num_x_minus, fit_change_num_y_minus = reset_direction_count()
                if out_time_count_flag:
                    if direction == change_direction:      # 后续出现该方向的其他点也归并到这一段里面
                        out_time_count = 0
                        move_cur.clear()
                        fit_change_num_x_plus, fit_change_num_y_plus, fit_change_num_x_minus, fit_change_num_y_minus = reset_direction_count()
                    else:
                        out_time_count += 1
                        move_cur.append(track_point)  # 被移除的轨迹点——下一段轨迹的开头
            if out_time_count>= out_time_count_th:
                update_flag = True
                track_cur = track_cur[:-out_time_count_th]
                # 确定位置
                if 'y' in direction:
                    position = find_position(track_cur[-1],box_buffer,box_in_buffer)
                else:
                    position = 'up'
                # 拟合
                if position:
                    func = match_dict['position']['direction']['func']
                    end_point = match_dict['position']['direction']['end_point']
                    need_length = match_dict['position']['direction']['need_length']
                    k_track_fit, b_track_fit, track_true_length, track_minpoint, track_maxpoint = func(np.array(track_cur))
                    (track_x_min, track_y_min) = track_minpoint
                    (track_x_max, track_y_max) = track_maxpoint
                    track_minpoint_pro, track_maxpoint_pro, track_length_pro = projection_length(track_minpoint,
                                                                                                 track_maxpoint,
                                                                                                 k_fit, b_fit)
                    print("真实轨迹长度：", track_true_length, "投影长度：", track_length_pro)
                    cv2.line(img_copy, track_minpoint, track_maxpoint, (255, 255, 0), 2)

                # 更新
                if update_flag:
                    if direction == 'x_plus':
                        match_dict[position][direction]['end_point'] = track_x_max
                    elif direction == 'x_minus':
                        match_dict[position][direction]['end_point'] = track_x_min
                    elif direction == 'y_plus':
                        match_dict[position][direction]['end_point'] = track_y_max
                    elif direction == 'y_minus':
                        match_dict[position][direction]['end_point'] = track_y_min
                    match_dict[position][direction]['need_length'] = match_dict[position][direction][
                                                                         'need_length'] - track_length_pro  # 同个方向的都拼一块了，没有判断起点终点！！！！！
                    if match_dict[position][direction]['need_length'] <= 0:
                        match_dict[position][direction]['need_length'] = match_dict_start[position][direction][
                            'need_length']
                        match_dict[position][direction]['end_point'] = match_dict_start[position][direction]['end_point']
                        done_dict[position] += 1  # 画好一条边
                        print('Done '+position)
                        print(done_dict)
                # 清除
                track_all.append(track_cur)
                track_img_all.append(track_cur_img)
                track_direction_all.append(direction)
                move_cur_tmp = copy.deepcopy(move_cur)
                track_cur, track_cur_img,move_cur, update_flag, \
                position, direction, change_direction, in_time_count, out_time_count, \
                out_count, is_out_count_over, out_time_count_flag= reset()
                track_cur = track_cur+move_cur_tmp
                move_cur_tmp.clear()
        #---------------------------------------------------------------------------------------------------     模拟轨迹
        # scb_rand = screen_contours_best_rand[0].squeeze()
        # screen_contours_best_rand_part_xy, screen_contours_best_rand_part, screen_contours_best_rand_part_length = dividing_contours(scb_rand, box_buffer, box_in_buffer)
        # (screen_contours_best_rand_left_xy, screen_contours_best_rand_right_xy,
        #  screen_contours_best_rand_up_xy) = screen_contours_best_rand_part_xy
        # (screen_contours_best_rand_left, screen_contours_best_rand_right, screen_contours_best_rand_up) = screen_contours_best_rand_part
        # (contour_rand_left_length, contour_rand_right_length, contour_rand_up_length) = screen_contours_best_rand_part_length
        # print("模拟轨迹周长：", contour_rand_left_length)
        # # 拟合曲线
        # k_rand_fit, b_rand_fit, track_rand_length, minpoint_rand, maxpoint_rand = fit_contours_vertical(screen_contours_best_rand_left_xy)
        # (left_x_rand_min, block_left_y_rand_min) = minpoint_rand
        # (left_x_rand_max, block_left_y_rand_max) = maxpoint_rand
        # print("模拟轨迹长度：", track_length)
        # cv2.line(img_copy, (left_x_rand_min, block_left_y_rand_min), (left_x_rand_max, block_left_y_rand_max), (0, 255, 0), 2)

        # ---------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------------     单张图片真实轨迹




        """
        track_xy = track_xy[(track_xy[:,0]>0) & (track_xy[:,1]>0)]

        track_t = tuple(np.array([list(np.expand_dims(track_xy, 1))]))

        cv2.drawContours(img_test, track_t, -1, (255,0, 0), 2)
        track_part_xy,track_part,track_part_length = dividing_contours(track_xy,box_buffer,box_in_buffer)
        (track_left_xy, track_right_xy, track_up_xy) = track_part_xy
        (track_left, track_right, track_up) = track_part
        (track_left_length, track_right_length, track_up_length) = track_part_length

        #去掉超出有效区域的轨迹点
        track_left_xy = track_left_xy[(track_left_xy[:, 0] >= xbuffer_left) & (track_left_xy[:, 0] <= x_inbuffer_left)]  # 左：去掉边框以外的点
        track_right_xy = track_right_xy[(track_right_xy[:, 0] >= x_inbuffer_right) & (track_right_xy[:, 0] <= xbuffer_right)]  # 右：去掉边框以外的点
        track_up_xy = track_up_xy[(track_up_xy[:, 1] >= ybuffer_up) & (track_up_xy[:, 1] <= y_inbuffer_up)]  # 上：去掉边框以外的点
        track_left, track_left_length = contours_to_tuple_and_len(track_left_xy)
        track_right, track_right_length = contours_to_tuple_and_len(track_right_xy)
        track_up, track_length = contours_to_tuple_and_len(track_up_xy)

        track_divide_all = track_right+track_up+track_left
        # cv2.drawContours(img_test, track_divide_all, -1, (0, 255, 255), 2)

        print("真实轨迹周长：", track_left_length)
        #拟合曲线
        k_track_fit, b_track_fit, track_true_length,track_minpoint,track_maxpoint = fit_contours_vertical(track_left_xy)
        (track_left_x_min, track_left_y_min) = track_minpoint
        (track_left_x_max, track_left_y_max) = track_maxpoint
        track_minpoint_pro, track_maxpoint_pro, track_length_pro = projection_length(track_minpoint,track_maxpoint, k_fit, b_fit)
        print("左真实轨迹长度：", track_true_length,"左投影长度：",track_length_pro)
        cv2.line(img_copy,  track_minpoint, track_maxpoint, (255, 255, 0), 2)

        k_track_fit_r, b_track_fit_r, track_true_length_r,track_minpoint_r,track_maxpoint_r = fit_contours_vertical(track_right_xy)
        track_minpoint_r_pro, track_maxpoint_r_pro, track_length_r_pro = projection_length(track_minpoint_r, track_maxpoint_r,
                                                                                     k_fit_r, b_fit_r)
        print("右真实轨迹长度：", track_true_length_r,"右投影长度：",track_length_r_pro)
        cv2.line(img_copy,  track_minpoint_r, track_maxpoint_r, (255, 255, 0), 2)

        k_track_fit_u, b_track_fit_u, track_true_length_u,track_minpoint_u,track_maxpoint_u = fit_contours_horizontal(track_up_xy)
        track_minpoint_u_pro, track_maxpoint_u_pro, track_length_u_pro = projection_length(track_minpoint_u, track_maxpoint_u,
                                                                                     k_fit_u, b_fit_u)
        print("上真实轨迹长度：", track_true_length_u,"上投影长度：",track_length_u_pro)
        cv2.line(img_copy,  track_minpoint_u, track_maxpoint_u, (255, 255, 0), 2)
        """
        # ---------------------------------------------------------------------------------------------------

        #  显示结果
        kernel_5x5 = np.ones((5, 5), np.uint8)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        mask_test = cv2.dilate(gray_mask, kernel_5x5, iterations=6)
        # opening = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel_5x5, iterations=10)

        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        combine = cv2.addWeighted(img_copy, 0.7, buffer_area_img, 0.3,0.8)
        combine_test = cv2.addWeighted(img_test, 0.7, buffer_area_img, 0.3, 0.8)
        cv2.circle(combine,track_point,20,(0,0,255),thickness=-1)
        fig = plt.figure()
        info = plt.imshow(combine, cmap='gray')
        print(i,image_path,'面积比：',area_per,'最大面积比:',area_per_min,'前后帧重心差距:',dist)
        # plt.pause(wait_time)
        # plt.show()
        plt.savefig(os.path.join(save_root, image_path))
        plt.close('all')
        i+=1

        # cv2.namedWindow("Mask Image", 0)
        # cv2.resizeWindow("Mask Image", 1920, 1080)  # 设置窗口大小
        # cv2.imshow('Mask',img_test)

        # cv2.namedWindow("Segmented Image", 0)
        # cv2.resizeWindow("Segmented Image", 1920, 1080)  # 设置窗口大小
        # cv2.imshow('Segmented Image',img_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if break_for:
            break
    print(done_dict)
        # 新的SN进来，轨迹置零，边框重置