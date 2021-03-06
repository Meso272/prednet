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
ysize=128


#categories = ['city', 'residential', 'road']

# Recordings used for validation and testing.
# Were initially chosen randomly such that one of the city recordings was used for validation and one of each category was used for testing.
#val_recordings = [('city', '2011_09_26_drive_0005_sync')]
#test_recordings = [('city', '2011_09_26_drive_0104_sync'), ('residential', '2011_09_26_drive_0079_sync'), ('road', '2011_09_26_drive_0070_sync')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)
train_snapshot_num=100
train_num=4*4*height*train_snapshot_num
val_snapshot_num=20
val_num=4*4*height*val_snapshot_num
test_snapshot_num=30
test_num=4*4*height*test_snapshot_num

train_start=1400
val_start=1500
test_start=1520
path="./aramco/"
train=np.zeros((train_num,xsize,ysize,1),dtype=np.float32)
train_source=[(0,0,0)]*train_num
for i in range(0,train_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+train_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((length,width,height))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,xsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                train[count*train_snapshot_num+i]=pict
                train_source[count*train_snapshot_num+i]=(x,y,z)
                count=count+1
                #print(array[x:x+size,y:y+size])


val=np.zeros((val_num,xsize,ysize,1),dtype=np.float32)
val_source=[0]*val_num
for i in range(0,val_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+val_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((length,width,height))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,xsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                val[count*val_snapshot_num+i]=pict
                val_source[count*val_snapshot_num+i]=(x,y,z)
                count=count+1

test=np.zeros((test_num,xsize,ysize,1),dtype=np.float32)
test_source=[0]*test_num
for i in range(0,test_snapshot_num):

    filename= "aramco-snapshot-%s.f32" % str(i+test_start)
    filepath=os.path.join(path,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((length,width,height))
        #print(array)
    count=0
    for z in range(0,height):
        for x in range(0,length,xsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=np.expand_dims(pict,2)
                test[count*val_snapshot_num+i]=pict
                test_source[count*val_snapshot_num+i]=(x,y,z)
                count=count+1

maximum=max(np.max(train),np.max(val),np.max(test))
minimum=min(np.min(train),np.min(val),np.min(test))
with open("minmax.txt","w") as f:
    f.write( str( (maximum,minimum) ) )

train=(train-minimum)/(maximum-minimum)
#print(val)
val=(val-minimum)/(maximum-minimum)
#print(val)
#test=(test+minimum)/(minimum+maximum)

hkl.dump(train, os.path.join(DATA_DIR, 'X_train.hkl'))
hkl.dump(train_source, os.path.join(DATA_DIR, 'sources_train.hkl'))
hkl.dump(val, os.path.join(DATA_DIR, 'X_val.hkl'))
hkl.dump(val_source, os.path.join(DATA_DIR, 'sources_val.hkl'))
#hkl.dump(test, os.path.join(DATA_DIR, 'X_test.hkl'))
#hkl.dump(test_source, os.path.join(DATA_DIR, 'sources_test.hkl'))
# Download raw zip files by scraping KITTI website
