
# import cv2
# import numpy as np
# # 读取照片
# img=cv2.imread('D:/Background_Green.JPEG')
 
# # 图像缩放
# img = cv2.resize(img,None,fx=0.5,fy=0.5)
# rows,cols,channels = img.shape
# print(rows,cols,channels)
# cv2.imshow('img',img)
 
# # 图片转换为灰度图
# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv',hsv)
 
# # 图片的二值化处理
# lower_blue=np.array([50,100,100])
# upper_blue=np.array([60,255,255])
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('mask',mask)
 
 
# #腐蚀膨胀
# erode=cv2.erode(mask,None,iterations=1)
# cv2.imshow('erode',erode)
 
# dilate=cv2.dilate(erode,None,iterations=1)
# cv2.imshow('dilate',dilate)
 
# #遍历替换
# for i in range(rows):
#  for j in range(cols):
#   if erode[i,j]==255: # 像素点为255表示的是白色，我们就是要将白色处的像素点，替换为红色
#    img[i,j]=(0,0,255) # 此处替换颜色，为BGR通道，不是RGB通道
# cv2.imshow('res',img)
 
# # 窗口等待的命令，0表示无限等待
# cv2.waitKey(0)

import cv2
import numpy as np
 
def cvtBackground(path,color):
    """
        功能：给证件照更换背景色（常用背景色红、白、蓝）
        输入参数：path:照片路径
                color:背景色 <格式[B,G,R]>
    """
    im=cv2.imread(path)
    im_hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)  #BGR和HSV的转换使用 cv2.COLOR_BGR2HSV
    #aim=np.uint8([[im[0,0,:]]])
    #hsv_aim=cv2.cvtColor(aim,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(im_hsv,np.array([im_hsv[0,0,0]-5,100,100]),np.array([im_hsv[0,0,0]+5,255,255]))  #利用cv2.inRange函数设阈值，去除背景部分
    mask1=mask                                                                                        #在lower_red～upper_red之间的值变成255
    img_median = cv2.medianBlur(mask, 5)  #自己加，中值滤波，去除一些边缘噪点
    mask = img_median
    mask_inv=cv2.bitwise_not(mask)  
    img1=cv2.bitwise_and(im,im,mask=mask_inv)   #将人物抠出
    bg=im.copy()
    rows,cols,channels=im.shape
    bg[:rows,:cols,:]=color
    img2=cv2.bitwise_and(bg,bg,mask=mask)       #将背景底板抠出
    img=cv2.add(img1,img2)
    image={'im':im,'im_hsv':im_hsv,'mask':mask1,'img':img,'img_median':img_median}
    cv2.startWindowThread() #加了这个后在图片窗口按Esc就可以关闭图片窗口
    for key in image:
        cv2.namedWindow(key)
        cv2.imshow(key,image[key])
    img = cv2.resize(img,None,fx=0.5,fy=0.5)
    cv2.imshow('red',img)
    cv2.imwrite('D:/Background_red.JPEG', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
#test
if __name__=='__main__':
    img=cvtBackground('D:/Background_Green.JPEG',[0,0,180])

    