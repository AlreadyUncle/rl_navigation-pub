# import matplotlib.pyplot as plt 
# import matplotlib.image as mpimg
import cv2 as cv
import numpy as np
import os
def get_obstacle_positions(map_limit, map_name):
#Load the file which contains all the obstacle positions
    return list(np.load(os.path.join('/Users/weishiqing/Desktop/rl_navigation_src/rl-navigation','maps', map_name+'.npy')))

# map_size = 10.0
# obs_pos = get_obstacle_positions(map_size,'train_map')
# print(len(obs_pos))
# print(obs_pos[0])

img_size = 1000
line_width = 5
img = np.zeros((img_size, img_size), dtype=np.uint8)
img.fill(255)
img[0:img_size, img_size-line_width:img_size] = 0
img[0:img_size, 0:line_width] = 0
img[img_size-line_width:img_size, 0:img_size] = 0
img[0:line_width, 0:img_size] = 0


win_name = "map"
# cv.namedWindow("map",cv.WINDOW_AUTOSIZE)
# cv.imshow(win_name, img)
cv.imwrite("/Users/weishiqing/Desktop/rl_navigation_src/rl-navigation/maps/test.png",img)

img_all = np.concatenate((img, img), axis=0)
img_all = np.concatenate((img_all, img_all), axis=1)
cv.imwrite("/Users/weishiqing/Desktop/rl_navigation_src/rl-navigation/maps/test_all.png",img_all)
