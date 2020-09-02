'''
Code for downloading and processing KITTI data (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

import os
#import requests
#from bs4 import BeautifulSoup
#import urllib.request
import numpy as np
#from imageio import imread
#from scipy.misc import imresize
import hickle as hkl
from aramco_settings import *
length=449
width=449
height=235
xsize=128
ysize=160


#categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
#val_recordings = [('city', '2011_09_26_drive_0005_sync')]
#test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
train_snapshot_num=80
train_num=4*3*height*train_snapshot_num
val_snapshot_num=10
train_num=4*3*height*val_snapshot_num
test_snapshot_num=10
test_num=4*3*height*test_snapshot_num

train_start=3501
val_start=3581
test_start=3591
path="./aramcp/"
train=np.zeros((train_num,xsize,ysize,1))
train_source=[0]*train_num
for i in range(0,train_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+train_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,zsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                train[count*train_snapshot_num+i]=pict
                train_source[count*train_snapshot_num+i]=i
                count=count+1
                #print(array[x:x+size,y:y+size])


val=np.zeros((val_num,xsize,ysize,1))
val_source=[0]*val_num
for i in range(0,val_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+val_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,zsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                val[count*val_snapshot_num+i]=pict
                val_source[count*val_snapshot_num+i]=i
                count=count+1

test=np.zeros((test_num,xsize,ysize,1))
test_source=[0]*test_num
for i in range(0,test_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+test_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((height,width))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,zsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                test[count*val_snapshot_num+i]=pict
                test_source[count*val_snapshot_num+i]=i
                count=count+1

maximum=np.max(train,val,test)
minimum=np.min(train,val,test)
with open("minmax.txt","w") as f:
    f.write(str(maximum)+"\n"+str(minimum))

train=(train+minimum)/(minimum+maximum)
val=(val+minimum)/(minimum+maximum)
test=(test+minimum)/(minimum+maximum)

hkl.dump(train, os.path.join(DATA_DIR, 'X_train.hkl'))
hkl.dump(train_source, os.path.join(DATA_DIR, 'sources_train.hkl'))
hkl.dump(val, os.path.join(DATA_DIR, 'X_val.hkl'))
hkl.dump(val_source, os.path.join(DATA_DIR, 'sources_val.hkl'))
hkl.dump(test, os.path.join(DATA_DIR, 'X_test.hkl'))
hkl.dump(test_source, os.path.join(DATA_DIR, 'sources_test.hkl'))
# Download raw zip files by scraping KITTI website
