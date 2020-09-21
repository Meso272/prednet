'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from aramco_settings import *
import argparse

def quantize(data,pred,error_bound):
    radius=32768
    diff = data - pred
    quant_index = (int) (abs(diff)/ error_bound) + 1
    #print(quant_index)
    if (quant_index < radius * 2) :
        quant_index =quant_index>> 1
        half_index = quant_index
        quant_index =quant_index<< 1
        #print(quant_index)
        quant_index_shifted=0
        if (diff < 0) :
            quant_index = -quant_index
            quant_index_shifted = radius - half_index
        else :
            quant_index_shifted = radius + half_index
        
        decompressed_data = pred + quant_index * error_bound
        #print(decompressed_data)
        if abs(decompressed_data - data) > error_bound :
            #print("b")
            return 0,data
        else:
            #print("c")
            data = decompressed_data
            return quant_index_shifted,data
        
    else:
        #print("a")
        return 0,data
    


parser = argparse.ArgumentParser(description='predictor')


parser.add_argument('--out', type=str, 
                    )

parser.add_argument('--error', type=float, default=1e-3)







args = parser.parse_args()

#n_plot = 40
#batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_aramco_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_aramco_model.json')
#test_file = os.path.join(DATA_DIR, 'X_test.hkl')
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))

predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
#X_test = test_generator.create_all()
test_start=1520
test_timestep_num=30
length=449
width=449
height=235
xsize=128
ysize=128
region_num=4*4*height
origfolder="./aramco/"
predfolder=args.out
maximum=0
minimum=0
with open("minmax.txt","r") as f:
    l=eval(f.read().strip())
    maximum=l[0]
    minimum=l[1]
    print(maximum)
    print(minimum)

series=np.zeros((region_num,nt,xsize,ysize,1))
source=[0]*region_num

series_start=test_start-nt
for i in range(0,nt):
    filename= "aramco-snapshot-%s.f32" % str(i+series_start)
    filepath=os.path.join(origfolder,filename)
    array=np.fromfile(filepath,dtype=np.float32).reshape((length,width,height))
    idx=0
    for z in range(0,height):
        for x in range(0,length,xsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=array[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=(np.expand_dims(pict,2)-minimum)/(maximum-minimum)
                series[idx][i]=pict
                source[idx]=(x,y,z)
                idx=idx+1
               
for i in range(0,test_timestep_num):

    filename= "aramco-snapshot-%s.f32" % str(i+test_start)
    qname="%sq.f32" % str(i+test_start)
    uname="%su.f32" % str(i+test_start)
    filepath=os.path.join(origfolder,filename)
    qpath=os.path.join(predfolder,qname)
    upath=os.path.join(predfolder,uname)
    preds=test_model.predict(series,batch_size=16)

    predarray=np.zeros((length,width,height),dtype=np.float32)
    for i,cor in enumerate(source):
        x=cor[0]
        y=cor[1]
        z=cor[2]

        endx=min(x+xsize,length)
        endy=min(y+ysize,width)
        #print(list(preds[i]))
        predarray[x:endx,y:endy,z]=preds[i,-1,:(endx-x),:(endy-y),0]
    predarray=predarray*(maximum-minimum)+minimum
    predarray.tofile(predpath)

    origarray=np.fromfile(filepath,dtype=np.float32).reshape((length,width,height))
    rng=np.max(origarray)-np.min(origarray)
    errorbound=rng*args.error
    qs=[]
    us=[]

    for i in range(length):
        for j in range(width):
            for k in range(height):
                q,u=quantize(origarray[i][j][k],predarray[i][j][k],errorbound)
                qs.append(q)
                if q==0:
                    us.append(u)
                else:
                    origarray[i][j][k]=u
    np.array(q,dtype=np.int32).tofile(qpath)
    np.array(u,dtype=np.float32).tofile(upath)
        #print(array)
    idx=0
    for z in range(0,height):
        for x in range(0,length,xsize):
            for y in range(0,width,ysize):
                endx=min(x+xsize,length)
                endy=min(y+ysize,width)
                pict=origarray[x:endx,y:endy,z]
                padx=xsize-pict.shape[0]
                pady=ysize-pict.shape[1]
                pict=np.pad(pict,((0,padx),(0,pady)))
                pict=(np.expand_dims(pict,2)-minimum)/(maximum-minimum)
                series[idx]=np.concatenate(( series[idx,1:,:,:,:],np.expand_dims(pict,axis=0) ),axis=0)
                idx=idx+1
               


'''
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))





# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

'''