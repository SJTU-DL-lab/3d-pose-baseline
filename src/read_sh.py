# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:10:56 2019

@author: jibin
"""
import numpy as np
import h5py
import data_utils
# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

SH_TO_GT_PERM = np.array([SH_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in SH_NAMES])
print(SH_TO_GT_PERM)
with h5py.File('Directions.54138969.h5','r') as h5f:
     poses = h5f['poses'][:]
     print(poses.shape)
    
     poses = poses[:,SH_TO_GT_PERM,:] #按照H36M的顺序来排列stacked hourglass的数据，这一步之后就已经是h3.6m的坐标系
     poses = np.reshape(poses,[poses.shape[0], -1])
     print(poses.shape)
     poses_final = np.zeros([poses.shape[0], len(H36M_NAMES)*2])#final的size是1612*64
     #*2是因为每个坐标的xy都放在一起即(x1,y1,x2,y2...)，且数据是二维的
     dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]* 2
     print(dim_to_use_x)
     dim_to_use_y = dim_to_use_x+1

     dim_to_use = np.zeros(len(SH_NAMES)*2,dtype=np.int32)#size为32
     
     dim_to_use[0::2] = dim_to_use_x
     print(dim_to_use)
     dim_to_use[1::2] = dim_to_use_y
     print(dim_to_use)
     poses_final[:,dim_to_use] = poses#将pose填入poses_final当中
     print(poses_final.shape)
     data_mean, data_std,  dim_to_ignore, dim_to_use = data_utils.normalization_stats( poses_final, dim=2 )
     print(data_mean.shape,data_std.shape,dim_to_ignore, dim_to_use)
data= poses_final[ :, dim_to_use ]
mu = data_mean[dim_to_use]
stddev = data_std[dim_to_use]
data_out = np.divide( (data - mu), stddev )
print(data_out.shape)