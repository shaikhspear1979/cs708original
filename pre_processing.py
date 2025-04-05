#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Shows live thermal image video in Tk window
#

HOST = "localhost"
PORT = 4223
UID = "HkA" # Change XYZ to the UID of your Thermal Imaging Bricklet
#import pptk

#from tinkerforge.ip_connection import IPConnection
#from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

import math
import time
import cv2
import numpy as np
import json
import os

#try:
#    #from Tkinter import Tk, Canvas, PhotoImage, mainloop, Label # Python 2
#    #from Queue import Queue, Empty
#except:
#    from tkinter import Tk, Canvas, PhotoImage, mainloop, Label # Python 3
#    from queue import Queue, Empty

#from PIL import Image, ImageTk
import sys
import numpy
from numpy.linalg import inv
numpy.set_printoptions(threshold=sys.maxsize)

WIDTH  = 80
HEIGHT = 60
SCALE  = 5 # Use scale 5 for 400x300 window size (change for different size). Use scale -1 for maximized mode
#image_queue = Queue()


# Creates standard thermal image color palette (blue=cold, red=hot)
def get_thermal_image_color_palette():
    palette = []

    #The palette is gnuplot's PM3D palette.
    #See here for details: https://stackoverflow.com/questions/28495390/thermal-imaging-palette
    for x in range(256):
        x /= 255.0
        palette.append(int(round(255*math.sqrt(x))))                  # RED
        palette.append(int(round(255*pow(x, 3))))                     # GREEN
        if math.sin(2 * math.pi * x) >= 0:
            palette.append(int(round(255*math.sin(2 * math.pi * x)))) # BLUE
        else:
            palette.append(0)

    return palette

# Callback function for high contrast image
def cb_high_contrast_image(image):
    # Save image to queue (for loop below)
    global image_queue
    image_queue.put(image)

def on_closing(window, exit_queue):
    exit_queue.put(True)

def distance(x1_in,y1_in,z1,x2_in,y2_in,z2,matrixx,transfrom_matrx):
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[x1_in],[y1_in],[1]]) # y1_in and x1_in are pixal coordinates
    pixal_coo2=np.matrix([[x2_in],[y2_in],[1]]) # y1_in and x1_in are pixal coordinates
    depth1 = np.matrix([z1])
    depth2 = np.matrix([z2])
    rotateToCamera = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    #print(transfrom_matrx)
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    P1_cam_frame_3d=P1_cam_frame_3d*depth1
    P2_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo2)
    P2_cam_frame_3d=P2_cam_frame_3d*depth2

    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
    P2_cam_frame_4d = np.matrix([[P2_cam_frame_3d[0]],[P2_cam_frame_3d[1]],[P2_cam_frame_3d[2]],[1]])
    #matrixx = np.linalg.inv(matrixx)
    matrixx = np.dot(matrixx,rotateToCamera)
    localpoint1=np.dot(matrixx,P1_cam_frame_4d)
    localpoint2=np.dot(matrixx,P2_cam_frame_4d)

    #localpoint = localpoint/localpoint[3]
    #coods_list=[float(P1_cam_frame_3d[1,0]),float(P1_cam_frame_3d[0,0]),float(P1_cam_frame_3d[2,0])]
    coods_list1=[float(localpoint1[0,0]),float(localpoint1[1,0]),float(localpoint1[2,0])]
    coods_list2=[float(localpoint2[0,0]),float(localpoint2[1,0]),float(localpoint2[2,0])]


    distance_points= math.sqrt((localpoint1[2] - localpoint2[2])*(localpoint1[2] - localpoint2[2])+(localpoint1[0] - localpoint2[0])*(localpoint1[0] - localpoint2[0]) + (localpoint1[1] - localpoint2[1])*(localpoint1[1] - localpoint2[1]))
    #print(distance_points,distance_points1,distance_points2)
    #print('width is ',distance_points1,' m')
    return distance_points

def height_cal(x1_in,y1_in,z1,x2_in,y2_in,z2,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[y1_in],[x1_in],[1]])
    pixal_coo2=np.matrix([[y2_in],[x2_in],[1]])
    depth1 = np.matrix([-z1])
    depth2 = np.matrix([-z2])
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    #print('top step1 coordinates w.r.t camera coordinate frame',P1_cam_frame_3d)
    P1_cam_frame_3d = np.dot(P1_cam_frame_3d,depth1)
    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])

    localpoint=np.dot(matrixx,P1_cam_frame_4d)

    localpoint = localpoint/localpoint[3]

   # print('top x y depth',pixal_coo2,depth2)
    P2_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo2)
   # print('bottom step1 coordinates w.r.t camera coordinate frame',P2_cam_frame_3d)
    P2_cam_frame_3d = np.dot(P2_cam_frame_3d,depth2)

    P2_cam_frame_4d = np.matrix([[P2_cam_frame_3d[0]],[P2_cam_frame_3d[1]],[P2_cam_frame_3d[2]],[1]]) 
   # print('4 d world coordinate frame',P2_cam_frame_4d)
    localpoint2=np.dot(matrixx,P2_cam_frame_4d)
    #print('step1 coordinates w.r.t world coordinate frame',localpoint2)
    localpoint2 = localpoint2/localpoint2[3]
   # print('bottom coordinates w.r.t world coordinate frame',localpoint2)
    #print('------------------------------------------------------------------------------')
   #print('word point',localpoint)
    distance_points= math.sqrt((localpoint[2] - localpoint2[2])*(localpoint[2] - localpoint2[2]))
    distance_points1= math.sqrt((localpoint[0] - localpoint2[0])*(localpoint[0] - localpoint2[0]))
    distance_points2= math.sqrt((localpoint[1] - localpoint2[1])*(localpoint[1] - localpoint2[1])) # height axis
    #print(distance_points,distance_points1,distance_points2)
    ds=math.sqrt(distance_points*distance_points + distance_points1*distance_points1 + distance_points2*distance_points2)

    coods_list=[localpoint[0,0][0,0],localpoint[1,0][0,0],localpoint[2,0][0,0]]
   # coods_list1=[localpoint1[0,0][0,0],localpoint1[1,0][0,0],localpoint1[2,0][0,0]]
    return distance_points2,localpoint,localpoint2,coods_list


from numpy.linalg import inv

'''def convert_coords_2d_to_3d(x1_in,y1_in,z1,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[y1_in],[x1_in],[1]])
 
    depth1 = np.matrix([-z1])
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    P1_cam_frame_3d = np.dot(P1_cam_frame_3d,depth1)
    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
 
    localpoint=np.dot(matrixx,P1_cam_frame_4d)

    localpoint = localpoint
    return localpoint 
'''



def load_command(number):
    f = open("./images/command/command"+str(number)+".txt", "r")
    cmd=f.read()[:-1]
    dictionary ={"translation":{ "en": str(cmd),
    "con": "con:faldstool_color == wardrobe_color"}
    }  
    # Serializing json 
    json_object = json.dumps(dictionary, indent = 4)
      
    # Writing to sample.json
    #with open(os.getcwd()+'/my_constraint.json', "w") as outfile:
    #    outfile.write(json_object)
    return cmd

def convert_coords_2d_to_3d(x1_in,y1_in,z1,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[x1_in],[y1_in],[1]]) # y1_in and x1_in are pixal coordinates
    depth1 = np.matrix([z1])
    rotateToCamera = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    #print(transfrom_matrx)
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    P1_cam_frame_3d=P1_cam_frame_3d*depth1
    P1_cam_frame_3d = np.array(P1_cam_frame_3d, dtype=np.float64)


    #print(P1_cam_frame_3d)
    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
    
    #matrixx = np.linalg.inv(matrixx)
    matrixx = np.dot(matrixx,rotateToCamera)
    localpoint=np.dot(matrixx,P1_cam_frame_4d)

    #localpoint = localpoint/localpoint[3]
    #coods_list=[float(P1_cam_frame_3d[1,0]),float(P1_cam_frame_3d[0,0]),float(P1_cam_frame_3d[2,0])]
    coods_list=[float(localpoint[0,0]),float(localpoint[1,0]),float(localpoint[2,0])]

    #print(coods_list)
    return coods_list



def is_nearly_parallel(v1, v2, threshold=0.01):
    # Check if two vectors are nearly parallel
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2)
    #print(cos_angle)
    return abs(cos_angle) > (1 - threshold)

def longest_distance(points, p1, p2):
    # Calculate perpendicular direction
    perpendicular_dir=np.array(p1) - np.array(p2)
    longest_dist = 0
    point_pair = []
    
    # Check pairs of points
    n = len(points)
    tH=1
    for i in range(n):
        for j in range(i + 1, n):
            new_line_vector = np.array(points[j]) - np.array(points[i])
            if(abs(points[i][1]) < tH  and abs(points[j][1]) < tH):
              #peint()
              if is_nearly_parallel(new_line_vector, perpendicular_dir):
                dist = np.linalg.norm(new_line_vector)
                if dist > longest_dist:
                    longest_dist = dist
                    point_pair=[points[i], points[j]]
    return point_pair, longest_dist

def convert_coords_3d_to_2d(x1_in,y1_in,z1_in,matrixx,transfrom_matrx):
    forcal_length=1447.8619
    from numpy.linalg import inv, det
    #calculate determinant of matrix
    deee=det(matrixx)
    #print('determinanet of the matrix is',deee)
    rotateToCamera = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    matrixx = np.dot(matrixx,rotateToCamera)
    inv_matrixx = inv(matrixx)
    
    inv_transfrom_matrx = inv(transfrom_matrx)

    P1_cam_frame_4d = np.matrix([[x1_in],[y1_in],[z1_in],[1]]) 

    P1_cam_frame_4d = np.dot(inv_matrixx,P1_cam_frame_4d)
    
   
    P1_cam_frame_3d = np.matrix([[float(P1_cam_frame_4d[0])/float(P1_cam_frame_4d[2])],[float(P1_cam_frame_4d[1])/float(P1_cam_frame_4d[2])],[1]])
    #print(P1_cam_frame_3d)
    #print()
    P1_cam_frame_3d = np.dot(inv_transfrom_matrx,P1_cam_frame_3d)
   # a=np.matrix([[0,-1,0],[1,0,0],[0,0,1]])
   # P1_cam_frame_3d=np.dot(a,P1_cam_frame_3d)

   # print('step2  coordinates w.r.t camera coordinate frame',P1_cam_frame_3d)
   # print(P1_cam_frame_3d,matrix[0])
 
    localpoint=P1_cam_frame_3d


    #localpoint1=np.dot(matrixx1,P1_cam_frame_4d)
   # print('step1 coordinates w.r.t world coordinate frame',localpoint)

    localpoint = localpoint
   
    return localpoint  

def pointcloud_list(x1_in,y1_in,z1,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[y1_in],[x1_in],[1]])
    depth1 = np.matrix([-z1])
   # matrixx = inv(matrixx)
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    #print('top step1 coordinates w.r.t camera coordinate frame',P1_cam_frame_3d)
    P1_cam_frame_3d = np.dot(P1_cam_frame_3d,depth1)
    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
    localpoint=np.dot(matrixx,P1_cam_frame_4d)

    localpoint = localpoint/localpoint[3]
    coods_list=[localpoint[0,0][0,0],localpoint[1,0][0,0],localpoint[2,0][0,0]]

    #print(coods_list)
    return coods_list

def pointcloud_list_without_global_transformation(x1_in,y1_in,z1,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[x1_in],[y1_in],[1]]) # y1_in and x1_in are pixal coordinates
    depth1 = np.matrix([-z1])
    #print(transfrom_matrx)
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    P1_cam_frame_3d=P1_cam_frame_3d*depth1

    #----viewmatrix-----
    viewMatrix=np.matrix([[0.9999765, 0.0052688005, 0.0044134087, 0.0], 
    [-0.006588121, 0.9177619, 0.39707687, 0.0], 
    [-0.001958339, -0.3970966, 0.91777486, 0.0], 
    [-0.0003428041, 0.00032242388, 4.321663e-05, 1.0000001]])
    #print(viewMatrix.shape)
    #----eular angles-----
    #Optional(SIMD3<Float>(-0.40832955, 0.004808777, -1.5779746))

    #-----transform matrix----
    tfm=np.matrix([[-0.0052687395, -0.9177616, 0.39709646, 0.0], 
        [0.99997634, -0.006588065, -0.0019583625, 0.0], 
        [0.0044134078, 0.39707676, 0.9177747, 0.0],
         [0.00034090638, -0.0003153269, 8.7698885e-05, 0.9999998]])
    #print(tfm.shape)

    #print(final_3dcoords[0,0])
    #print('depth1',depth1)
    #print('final coordinates areSS',final_3dcoords)
    #print('top step1 coordinates w.r.t camera coordinate frame',P1_cam_frame_3d)
    #P1_cam_frame_3d = np.dot(P1_cam_frame_3d,depth1)
    #P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
    #localpoint=np.dot(viewMatrix,P1_cam_frame_4d)

    #localpoint = localpoint/localpoint[3]
    coods_list=[float(P1_cam_frame_3d[1,0]),float(P1_cam_frame_3d[0,0]),float(P1_cam_frame_3d[2,0])]
    #coods_list=[float(P1_cam_frame_4d[1,0]),float(P1_cam_frame_4d[0,0]),float(P1_cam_frame_4d[2,0])]

    #print(coods_list)
    return coods_list



def pointcloud_list_without_global_transformation_nointrinsic(x1_in,y1_in,z1,matrixx,transfrom_matrx):
    forcal_length=1447.8619
   # transfrom_matrx=np.matrix([[0.0006876147,0.0,0.0],[0.0,0.0006876147,0.0],[-0.64709806,-0.5007238,1]])
    pixal_coo1=np.matrix([[y1_in],[x1_in],[1]]) # y1_in and x1_in are pixal coordinates
    depth1 = np.matrix([z1])
    rotateToCamera = np.matrix([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    #print(transfrom_matrx)
    P1_cam_frame_3d = np.dot(transfrom_matrx,pixal_coo1)
    P1_cam_frame_3d=P1_cam_frame_3d*depth1

    P1_cam_frame_4d = np.matrix([[P1_cam_frame_3d[0]],[P1_cam_frame_3d[1]],[P1_cam_frame_3d[2]],[1]])
    #matrixx = np.linalg.inv(matrixx)
    matrixx = np.dot(matrixx,rotateToCamera)
    localpoint=np.dot(matrixx,P1_cam_frame_4d)

    #localpoint = localpoint/localpoint[3]
    #coods_list=[float(P1_cam_frame_3d[1,0]),float(P1_cam_frame_3d[0,0]),float(P1_cam_frame_3d[2,0])]
    coods_list=[float(localpoint[1,0]),float(localpoint[0,0]),float(localpoint[2,0])]

    #print(coods_list)
    return coods_list
def depth_image_generation(path):
    #print(path)
    f = open(path, "r")
    a=0
    depth_image = np.zeros((192,256))
    #print(depth_image)
    width =0
    height = 0
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
        #print(height,width,width*height)
        #print(float(x)*20)
        depth_image[height][width] = float(x)
        width+=1
        if(width == 256):
            width=0
            height+=1
    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
    #depth_image = cv2.resize(depth_image, (int(1440),int(1920)), interpolation = cv2.INTER_AREA)
    #cv2.imshow('depth',depth_image)
    #cv2.waitKey(0)
    return depth_image

def matrix_genertion(path):
    f = open(path, "r")
    a=0
    matrix = np.zeros((4,4))
    #print(depth_image)
    width =0
    height = 0
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
        #print(height,width,width*height)
        #print(float(x)*20)
        matrix[width][height] = float(x)
        width+=1
        if(width == 4):
            width=0
            height+=1
    #print('transform matrix is',matrix)
    #cv2.imshow('depth',depth_image)
    #cv2.waitKey(0)
    return matrix


def ar_object_dimention(path):
  with open(path, 'r') as file:
    line = file.readlines()  # Read the first line from the file and remove any leading/trailing whitespace
    # Extract the width and height values
    width = float(line[0])
    height = float(line[1])
    depth = float(line[2])

  return width,height,depth

def euler_genertion(path):
    #print(path)
    f = open(path, "r")
    angles=[]
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
      angles.append(float(x))
    return angles

def touchpoints_genertion(path):
    #print(path)
    f = open(path, "r")
    angles=[]
    dd=[]
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for p in range(1,5):
      x = 0+(p-1)*2
      y = 1+(p-1)*2
      for j in f:
        dd.append(j)
      angles.append([int(dd[x]),int(dd[y])])
    return angles

def trans_vector_genertion(path):
   # print(path)
    f = open(path, "r")
    angles=[]
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
      angles.append(float(x))
    return angles

def translate_matrix_genertion(path):
    #print(path)
    f = open(path, "r")
    a=0
    matrix = np.zeros((4,4))
    #print(depth_image)
    width =0
    height = 0
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
        #print(height,width,width*height)
        #print(float(x)*20)
        matrix[height][width] = float(x)
        height+=1
        if(height == 4):
            height=0
            width+=1
    #matrix[3][0]=0
    #matrix[3][1]=0
    #matrix[3][2]=0
    #matrix[3][3]=1
    #print('matrix is',matrix)
    #cv2.imshow('depth',depth_image)
    #cv2.waitKey(0)
    return matrix


def intric_matrix_genertion(path):
    #print(path)
    f = open(path, "r")
    a=0
    matrix = np.zeros((3,3))
    #print(depth_image)
    width =0
    height = 0
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
        #print(height,width,width*height)
        #print(float(x)*20)
        matrix[width][height] = float(x)
        width+=1
        if(width == 3):
            width=0
            height+=1
    #matrix[3][0]=0
    #matrix[3][1]=0
    #matrix[3][2]=0
    #matrix[3][3]=1
    #print('matrix is',matrix)
    #cv2.imshow('depth',depth_image)
    #cv2.waitKey(0)
    return matrix


def intric_viewMatrix_genertion(path):
    #print(path)
    f = open(path, "r")
    a=0
    matrix = np.zeros((4,4))
    #print(depth_image)
    width =0
    height = 0
    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
    for x in f:
        #print(height,width,width*height)
        #print(float(x)*20)
        matrix[height][width] = float(x)
        height+=1
        if(height == 4):
            height=0
            width+=1
    #matrix[3][0]=0
    #matrix[3][1]=0
    #matrix[3][2]=0
    #matrix[3][3]=1
    #print('matrix is',matrix)
    #cv2.imshow('depth',depth_image)
    #cv2.waitKey(0)
    return matrix

# depth image szie 256*192
#normal imahe side 1920,1440
if __name__ == "__main__":
    
    # Create Tk window and label
    original_image = cv2.imread('./images/image/depthimage0.png',1)
#    original_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
#    print(original_image.shape)
#    original_image = cv2.resize(original_image, (int(1920),int(1440)), interpolation = cv2.INTER_AREA)
#    #img2= cv2.imread('./images/image0.png',1)
#
#    f = open('./images/depth/depth0.txt', "r")
#    a=0
#    depth_image = np.zeros((192,256))
#    #print(depth_image)
#    width =0
#    height = 0
#    # this loop is to create the depth image matrix using list of coordinates of 1,x.lenth list
#    for x in f:
#        #print(height,width,width*height)
#        depth_image[height][width] = x
#        width+=1
#        if(width == 256):
#            width=0
#            height+=1
#        
#    #print(depth_image)
#    x1_in=540
#    y1_in=500
#    x2_in=1400
#    y2_in=500
#    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
#    depth_image = cv2.resize(depth_image, (int(1920),int(1440)), interpolation = cv2.INTER_AREA) #(column,row)
#    print(depth_image.shape)
#    depth_pixel_value1=depth_image[y1_in-1][x1_in-1]
#    depth_pixel_value2=depth_image[y2_in-1][x2_in-1]
#    print('depth_pixel_value1',depth_pixel_value1)
#    print('depth_pixel_value2',depth_pixel_value2)
#    center_coordinates1=(x1_in,y1_in)
#    center_coordinates2=(x2_in,y2_in)
#    radius=2
#    color=[0,0,0]
#    color2=[0,0,255]
#
#    imaoriginal_imagege = cv2.circle(original_image, center_coordinates1, radius, color2, 2)
#    imaoriginal_imagege = cv2.circle(original_image, center_coordinates2, radius, color2, 2)
#    depth_image = cv2.circle(depth_image, center_coordinates1, radius, color, 2)
#    depth_image = cv2.circle(depth_image, center_coordinates2, radius, color, 2)
#  #  print('depth_pixel_value2',depth_pixel_value2)
#    
#
#    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#   # img2 = cv2.resize(img2, (80*5,60*5), interpolation = cv2.INTER_AREA)
#    #
#    cv2.imshow('depth image',depth_image)
#    cv2.imshow('original image',original_image)
#    cv2.waitKey(0)
#
#    distance(x1_in,y1_in,depth_pixel_value1,x2_in,y2_in,depth_pixel_value2)
#
#    #for i in img:
#        #if i[0][0] == 2:
#
#    
#
#        
#        #window.update()
#        
   # window.destroy()

