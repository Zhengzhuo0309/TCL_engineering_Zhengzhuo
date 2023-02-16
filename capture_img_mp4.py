# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/1 18:54
@Auth ： pan.xie
@File ：capture_img_url.py
@IDE ：PyCharm
"""
import cv2

cap = cv2.VideoCapture("2023.mp4")
isOpened = cap.isOpened()# 判断是否打开‘
print("isOpened = ",isOpened)
fps = cap.get(cv2.CAP_PROP_FPS)# 帧率 每秒展示多少张图片
print("fps = ",fps)
i = 0
while(cap.isOpened()):
    print(i)
    i = i+1
    ret, frame = cap.read()
    print("frame = ",frame)


    frame = cv2.resize(frame, (1280,768))
    print(frame.shape)
    cv2.imshow('frame',frame)
    cv2.imwrite("./data/{}.jpg".format(i), frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
