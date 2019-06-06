#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path      = "./docs/images/road.jpeg"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
# 缩放并且补充边缘部分，变成正方形
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
# 增加一个新的维度，即变成(n,c,h,w)
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

print(type(pred_sbbox), pred_sbbox.shape)
print(type(pred_mbbox), pred_mbbox.shape)
print(type(pred_lbbox), pred_lbbox.shape)
'''
输出结果：
<class 'numpy.ndarray'> (1, 52, 52, 3, 85)
<class 'numpy.ndarray'> (1, 26, 26, 3, 85)
<class 'numpy.ndarray'> (1, 13, 13, 3, 85)
'''

input_size
# 先把数组变成 (-1,85), 再
pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
print("pred_bbox : ")
print(type(pred_bbox), pred_bbox.shape)
'''
输出结果：
<class 'numpy.ndarray'> (10647, 85)
'''

bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
print("bboxes : ")
print(type(bboxes), bboxes.shape)
'''
输出结果：
<class 'numpy.ndarray'> (113, 6)
'''

bboxes = utils.nms(bboxes, 0.45, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()




