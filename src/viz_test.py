import numpy as np
import viz
import matplotlib.pyplot as plt 
#matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D

#z这个维度说明的是数据的维度，可能是16或者17
dim=17

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

# Stacked Hourglass produces 16 joints. These are the names.  10,12,14,16,11,13,15,9,2,0,1,3,5,7,4,6,8
H_NAMES = ['']*17
H_NAMES[16] = 'RFoot'
H_NAMES[14] = 'RKnee'
H_NAMES[12] = 'RHip'
H_NAMES[11] = 'LHip'
H_NAMES[13] = 'LKnee'
H_NAMES[15] = 'LFoot'
H_NAMES[10] = 'Hip'
H_NAMES[9]  = 'Spine'
H_NAMES[2]  = 'Thorax'
H_NAMES[1]  = 'Head'
H_NAMES[8] = 'RWrist'
H_NAMES[6] = 'RElbow'
H_NAMES[4] = 'RShoulder'
H_NAMES[3] = 'LShoulder'
H_NAMES[5] = 'LElbow'
H_NAMES[7] = 'LWrist'
H_NAMES[0] = 'Neck/Nose'

enc_in=np.zeros([1, 17*2])
dec_out=np.zeros([1,17*3])
data=np.load("Ai_train.npy")
a=data[2:3,:]
x=a[:,0:17]
y=a[:,17:34]
xx=a[:,34:51]
yy=a[:,51:68]
zz=a[:,68:85]

#H_TO_GT_PERM = np.array([H_NAMES.index( h ) for h in H36M_NAMES if h != '' and h in H_NAMES])
H_TO_GT_PERM = np.array([10,12,14,16,11,13,15,9,2,0,1,3,5,7,4,6,8])
x=x[0,H_TO_GT_PERM]
y=y[0,H_TO_GT_PERM]
xx=xx[0,H_TO_GT_PERM]
yy=yy[0,H_TO_GT_PERM]
zz=zz[0,H_TO_GT_PERM]
#将xxxxyyyyy的顺序变为xyxyxy的顺序
for i in np.arange(17):
    enc_in[0,i*2]=x[i] #17*2
    enc_in[0,i*2+1]=y[i]
    dec_out[0,i*3]=xx[i]#17*3
    dec_out[0,i*3+1]=zz[i]
    dec_out[0,i*3+2]=-yy[i]


#先要得到H3.6m格式下使用的维度
if dim==16:
    dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
    dim_to_use_2 = np.zeros(16*2,dtype=np.int32)
    dim_to_use_x_3d = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 3
    dim_to_use_3 = np.zeros(16*3,dtype=np.int32)
else:
    dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 2
    dim_to_use_2 = np.zeros(17*2,dtype=np.int32)
    dim_to_use_x_3d = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
    dim_to_use_3 = np.zeros(17*3,dtype=np.int32)    
dim_to_use_y = dim_to_use_x+1
dim_to_use_2[0::2] = dim_to_use_x
dim_to_use_2[1::2] = dim_to_use_y
#3d时同上

dim_to_use_y_3d = dim_to_use_x_3d+1
dim_to_use_z_3d = dim_to_use_x_3d+2
dim_to_use_3[0::3] = dim_to_use_x_3d
dim_to_use_3[1::3] = dim_to_use_y_3d
dim_to_use_3[2::3] = dim_to_use_z_3d

enc_final = np.zeros([1, 32*2])
dec_final = np.zeros([1,32*3])
#接下来需要将第17个点删除（这里出了点问题）这里并不是真的要删除spine这个点，只是h3.6m中spine对应的就是MPII中的Neck/Nose
if dim==16:
    dim_x=np.where(np.array([x != '' and x != 'Spine' for x in H_NAMES]))[0]*2
    dim_2 = np.zeros(16*2,dtype=np.int32)
    dim_x_3d = np.where(np.array([x != '' and x != 'Spine' for x in H_NAMES]))[0]*3
    dim_3 = np.zeros(16*3,dtype=np.int32)
else:
    dim_x=np.where(np.array([x != '' for x in H_NAMES]))[0]*2
    dim_2 = np.zeros(17*2,dtype=np.int32)
    dim_x_3d = np.where(np.array([x != '' for x in H_NAMES]))[0]*3
    dim_3 = np.zeros(17*3,dtype=np.int32)
    
dim_y = dim_x+1
dim_2[0::2] = dim_x
dim_2[1::2] = dim_y


dim_y_3d = dim_x_3d+1
dim_z_3d = dim_x_3d+2
dim_3[0::3] = dim_x_3d
dim_3[1::3] = dim_y_3d
dim_3[2::3] = dim_z_3d


enc_final[:,dim_to_use_2] = enc_in[:,dim_2]
dec_final[:,dim_to_use_3]=dec_out[:,dim_3]
print(dec_out,'dec_out')
# Visualize random samples
import matplotlib.gridspec as gridspec

# 1080p	= 1,920 x 1,080
fig = plt.figure( figsize=(19.2, 10.8) ) #先画出图框大小

gs1 = gridspec.GridSpec(1, 2) # 5 rows, 9 columns
gs1.update(wspace=0.05, hspace=0.05) # set the spacing between axes.
plt.axis('off')

subplot_idx, exidx = 1, 1
nsamples = 1
for i in np.arange( nsamples ):

    # Plot 2d pose
    ax1 = plt.subplot(gs1[subplot_idx-1])
    p2d = enc_final[exidx-1,:]
    viz.show2Dpose( p2d, ax1 )
    ax1.invert_yaxis()

    # Plot 3d gt
    ax2 = plt.subplot(gs1[subplot_idx], projection='3d')
    p3d = dec_final[exidx-1,:]
    viz.show3Dpose( p3d, ax2 )

    exidx = exidx + 1
    subplot_idx = subplot_idx + 3

plt.show()
