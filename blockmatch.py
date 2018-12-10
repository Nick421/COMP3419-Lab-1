
# coding: utf-8

# In[ ]:


# Package required and global variable declaration
import numpy as np
import cv2
import os
import sys
import math

frame_save_path = './frames/'
outputframe_save_path = './demooutputframes/'
diffoutputframe_save_path = './diffoutputframes/'
path_to_video = './monkey.mov'
path_to_output_video = './demoMonkey.mov'

grid_size = 9
radius = 3

h = 0
w = 0

v0 = 120
v1 = 150

withBoundaryOption = False


# In[ ]:


def ssd(a, b):
    #check underflow
    return np.sqrt(np.sum((np.power((a-b),2))))


# In[ ]:


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[ ]:


def calc(frame1 , frame2, index):
    for y1 in range(h):
        #print("in y1", y1)
        i = y1*grid_size
        for x1 in range(w):
            #print("in x1")

            j = x1*grid_size
            block1 = frame1[i:i+grid_size, j:j+grid_size,:]

            radius_hood = []

            for y2 in range(y1-radius,y1+radius+1):
                    #print("in y2 ",y2)
                    i2 = y2*grid_size
                    if not (0 <= y2 < h):
                        continue

                    for x2 in range(x1-radius,x1+radius+1):
                        #print("in x2 ",y2)
                        j2 = x2*grid_size
                        if not (0 <= x2 < w):
                            continue

                        block2 = frame2[i2:i2+grid_size, j2:j2+grid_size,:]

                        # find SSD of current frame and neighbour in radius
                        ssd_current = ssd(block1, block2)
                        #print(ssd_current)
                        #print("before append ",radius_hood)
                        radius_hood.append((ssd_current, x2, y2))
                        #print("after append ",radius_hood)

            #takes the neigbour that has closet SSD
            #print("before finding min",radius_hood)
            ssdmin = min(radius_hood)
            #print("min ssd", ssdmin)
            if (v0 < ssdmin[0] < v1):
                #print("sdd: ",ssdmin[0]," x1: ",i," y1: ",j," x2: ",ssdmin[1]," y2: ",ssdmin[2] )
                if (withBoundaryOption == True):
                    diff = cv2.imread(diffoutputframe_save_path + 'frame%d.tif' %(index+1))
                    drawBound(diff,frame2)

                arrow(frame2,x1,y1,ssdmin[1],ssdmin[2])


    cv2.imwrite(outputframe_save_path + 'frame%d.tif' %index, frame2)


# In[ ]:


def difference(dilate, thres,index):
    diff = np.subtract(dilate,thres)
    cv2.imwrite(diffoutputframe_save_path + 'frame%d.tif' %index, diff)


# In[ ]:


def drawBound(diff,og):
    for y in range(diff.shape[0]):
        for x in range(diff.shape[1]):
            #print(diff[y][x])
            if np.any(diff[y][x] == 255.0):
                #cv2.circle(og,(x,y), 2, (0,255,0), 1)
                og[y,x] = [0,255,0]


# In[ ]:


def toBinary(frame):
    img_grey = frame
    img_new_grey = 0.212671* frame[:,:,2] + 0.715160* frame[:,:,0] + 0.072169* frame[:,:,1]
    img_grey[:,:,0] = img_new_grey
    img_grey[:,:,1] = img_new_grey
    img_grey[:,:,2] = img_new_grey

    return img_grey


# In[ ]:


def arrow(frame, x1, y1, x2 ,y2):

    pt1 = (x1*grid_size,y1*grid_size)
    pt2 = (x2*grid_size,y2*grid_size)
    #bgr
    cv2.arrowedLine(frame, pt1, pt2, (0,0,255), 2)


# In[ ]:


def img_threshold(img_grey):
    img_thres = ((img_grey > 100) + np.zeros(img_grey.shape)) * 255
    return img_thres


# In[ ]:


def dilateErode2D(img_in, kernel, type):

    assert type == 'dilate' or type == 'erosion'
    img = img_in[:,:,0]
    final = np.zeros(img_in.shape)
    newimg = np.copy(img)

    kernelSize = kernel.shape[0]
    radius = int(kernelSize / 2)

    h, w  = img.shape
    for x in range (radius, h-radius):
        for y in range (radius, w-radius):
            demo_array = img[x-radius: x+radius+1, y-radius: y+radius+1]
            if type == 'dilate':
                result = np.amax(demo_array * kernel)
            else:
                result = np.amin(demo_array * kernel)
            newimg[x][y] = result

    final[:, :, 0] = newimg
    final[:, :, 1] = newimg
    final[:, :, 2] = newimg

    return final


# In[ ]:


cap = cv2.VideoCapture(path_to_video)
create_dir_if_not_exists(frame_save_path) # Or you can create it manully
if not cap.isOpened():
    print('{} not opened'.format(path_to_video))
    sys.exit(1)

frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_counter = 0                                             # FRAME_COUNTER
while(1):
    return_flag, frame = cap.read()
    if not return_flag:
        print('Video Reach End')
        break
    # Main Content - Start
    cv2.imwrite(frame_save_path + 'frame%d.tif' % frame_counter, frame)
    frame_counter += 1
    # Main Content - End
cap.release()


# In[ ]:


#get dilated frames and threshold frames
#and get difference frames stored it
if (withBoundaryOption == True):
    kernel = np.ones((3, 3), np.uint8)
    index = 0
    create_dir_if_not_exists(diffoutputframe_save_path)
    while True:
        frame = cv2.imread(frame_save_path + 'frame%d.tif' %index)
        if frame is None:
            break
        img_grey = toBinary(frame)
        img_thres = img_threshold(img_grey)
        dilate = dilateErode2D(img_in=img_thres, kernel = kernel, type='dilate')
        difference(dilate,img_thres,index)
        index += 1


# In[ ]:


#SSD
h = int(frame_height//grid_size)
w = int(frame_width//grid_size)
#print(h)
#print(w)
index = 0
create_dir_if_not_exists(outputframe_save_path)

while index < 10:

    frame_1 = cv2.imread(frame_save_path + 'frame%d.tif' %index)
    frame_2 = cv2.imread(frame_save_path + 'frame%d.tif' %(index+1))

    if frame_1 is None or frame_2 is None:
        break
    calc(frame_1, frame_2,index)
    #print(index)
    index += 1
print('Finish!')


# In[ ]:


#merge export Video
out = cv2.VideoWriter(path_to_output_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (int(frame_width), int(frame_height)))
frame_counter = 0
while(1):
    img = cv2.imread(outputframe_save_path + 'frame%d.tif' % frame_counter)
    if img is None:
        print('No more frames to be loaded')
        break;
    out.write(img)
    #print(frame_counter)
    frame_counter += 1
out.release()
cv2.destroyAllWindows()
