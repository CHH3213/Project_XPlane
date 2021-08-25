1. 直接运行ddpg_train.py开始训练，暂定1000个episodes，每个episode200steps
   python ddpg_train.py
2. 环境为env2.py，重点关注下rewardCalcul(),step(),reset()这几个函数，我写的比较粗糙，有可能不对。

其中

> state:[roll_rate,pitch_rate,yaw_rate,row,pitch,yaw,altitude,airspeed]
>
> action：[elevator,aileron,rudder,throttle,speed_brake]

检查一下我写的action和state是否对应就是这些。

reward的设置，我现在设置的很粗糙，你们可以根据训练情况再调试。

