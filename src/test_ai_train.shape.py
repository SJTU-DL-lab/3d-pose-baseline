# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:16:18 2019

@author: jibin
"""
import numpy as np
import matplotlib.pyplot as plt 
#matplotlib notebook
from mpl_toolkits.mplot3d import Axes3D
a=np.load("Ai_train.npy")
a=a[0:1,:]
x=a[:,0:17]
print(x.shape)
y=a[:,17:34]
xx=a[:,34:51]
yy=a[:,51:68]
zz=a[:,68:85]
#画出三维坐标系：
ax = plt.subplot(projection='3d') 
for x1,y1,z1 in zip(xx,yy,zz):
    color='r'
    ax.scatter(x1,y1,z1,c=color,marker='*',s=160,linewidth=1,edgecolor='b')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')

plt.show()

