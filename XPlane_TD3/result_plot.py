# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:18:23 2016
used on  Wen May  12 13:18:23 2021

@author: steven
@modified: chh
"""
# import matplotlib.pyplot as plt
import numpy as np
# import main_TD3 as td3
# dir = './save/xplane_TD3_10/data/'
lines = np.loadtxt('xplane_TD3_pretrain6.txt', comments="#", delimiter="\n", unpack=False)
# lines = np.loadtxt('xplane_TD3_11.txt', comments="#", delimiter="\n", unpack=False)
# lines = np.loadtxt("episode_reward.txt", comments="#", delimiter="\n", unpack=False)
# a =np.zeros([5,1])
# b=a[:,np.newaxis]
# print(a.shape,b.shape)
# print(a)
# print(b)
plt.plot(lines)
plt.show()


# ## draw exercise
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# sns.set() # 设置美化参数，一般默认就好
#
# def get_data():
#     line1 = np.loadtxt('xplane_TD3_pretrain6.txt', comments="#", delimiter="\n", unpack=False)
#     line2 = np.loadtxt('xplane_TD3_pretrain4.txt', comments="#", delimiter="\n", unpack=False)
#     line3 = np.loadtxt('xplane_TD3_pretrain5.txt', comments="#", delimiter="\n", unpack=False)
#     return line1,line2,line3
#
# data = get_data()
# data = pd.DataFrame(data).melt(var_name='episode',value_name='reward')
# sns.lineplot(x='episode',y='reward',data=data)
# label=['algo1','algo2','algo3','algo4']
# df = []
# for i in range(len(data)):
#     df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='loss'))
#     df[i]['algo']=label[i]
#
# df = pd.concat(df)
# sns.lineplot(x='episode',y='loss',hue='algo',data=df,style='algo') # hue为调整颜色，style为调整线的类型
# sns.lineplot(x=range(len(lines)),y=lines)
# plt.show()