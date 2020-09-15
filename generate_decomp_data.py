
import argparse
import numpy as np
import os


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
                    help='output folder')
parser.add_argument('--error', type=float, 
                    help='error bound')
args = parser.parse_args()

start=1520
end=1550

for i in range(start,end):
    origpath="aramco/aramco-snapshot-%s.f32" % str(i)
    predpath="aramco_pred/aramco-snapshot-%s-pred.f32" % str(i)
    outname="aramco-snapshot-%s-decomp.f32" %str(i)
    outpath=os.path.join(args.out,outname)
    oarray=np.fromfile(origpath,dtype=np.float32)
    parray=np.fromfile(predpath,dtype=np.float32)
    rng=np.max(oarray)-np.min(oarray)
    curerror=args.error*rng
    for i in range(oarray.shape[0]):
        _,p=quantize(oarray[i],parray[i],curerror)
        parray[i]=p
    parray.tofile(outpath)

