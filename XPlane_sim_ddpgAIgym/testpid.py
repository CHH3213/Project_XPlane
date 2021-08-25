
import xpc as xp



class Pid():
    err = 0
    err_last = 0
    err_next = 0
    increase = 0
    count=0
    def __init__(self,set_val, Kp, Ki, Kd):
        self.KP = Kp
        self.KI = Ki
        self.KD = Kd
        self.set_val = set_val
    def pid_increment(self,now_val):
        self.err = self.set_val - now_val
        # print('error:',self.err)
        # if abs(self.err)>5:
        #     self.increase = self.KP * (self.err - self.err_next) + self.KD * (self.err - 2 * self.err_next + self.err_last)
        # else:
        self.increase = self.KP * (self.err - self.err_next) + self.KI * self.err + self.KD * (self.err - 2 * self.err_next + self.err_last)
        self.err_last = self.err_next
        # print('incease:',self.increase)
        self.err_next = self.err
        return self.increase

client=xp.XPlaneConnect('0.0.0.0', '127.0.0.1', 49009, 1, 1000)
drefs1=["sim/cockpit2/gauges/indicators/roll_vacuum_deg_pilot",
      "sim/cockpit2/gauges/indicators/pitch_vacuum_deg_pilot"]

drefs2=["sim/cockpit2/controls/yoke_roll_ratio",
        "sim/cockpit2/controls/yoke_pitch_ratio"]

roll_pid = Pid(0,0.1,0.01,0)
pitch_pid =Pid(0,0.06, 0.002, 0)
yokeroll_val=0
yokepitch_val=0
value=[0,0]
while True:
    client.sendCOMM('sim/instruments/copilot_DG_sync_mag')  # magnetic north.

    roll,pitch = client.getDREFs(drefs1)
    print("roll:",roll,"pitch:",pitch)

    yokeroll_val += roll_pid.pid_increment(roll[0])
    if abs(yokeroll_val) > 1:
        yokeroll_val = 1 if yokeroll_val > 0 else -1
    value[0]=yokeroll_val

    yokepitch_val += pitch_pid.pid_increment(pitch[0])
    if abs(yokepitch_val) > 1:
        yokepitch_val = 1 if yokepitch_val > 0 else -1
    value[1]=yokepitch_val

    client.sendDREFs(drefs2,value)



