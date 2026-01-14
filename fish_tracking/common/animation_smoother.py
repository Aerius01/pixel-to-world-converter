from fish_tracking.common.globals import *
import numpy as np
from fish_tracking.common.math_utils import get_euler_from_quaternion,get_quaternion_from_euler
class AnimationSmoother:
    def __init__(self):
        # current velocity for position
        self.cs_pos = np.array([0.0, 0.0, 0.0])
        # current velocity for cam euler angles
        self.cs_cam_angle = np.array([0.0, 0.0, 0.0])


    def update_cam_pose_animation(self,cp, ep, cco, eco):
        '''
        Desciption: Smooths the camera pose animation
        Input:
            cp: current position
            ep: expected position
            cco: current camera orientation
            eco: expected camera orientation
        Output:
            cp: new position
            cco: new camera orientation
        '''
        cp[0],self.cs_pos[0] = self.update_pos(cp[0], ep[0], self.cs_pos[0])
        cp[1],self.cs_pos[1] = self.update_pos(cp[1], ep[1], self.cs_pos[1])
        cp[2],self.cs_pos[2] = self.update_pos(cp[2], ep[2], self.cs_pos[2])

        cam_euler = get_euler_from_quaternion(cco)
        target_cam_euler = get_euler_from_quaternion(eco)

        cam_euler[0], self.cs_cam_angle[0] = self.update_angle(cam_euler[0], target_cam_euler[0], self.cs_cam_angle[0])
        cam_euler[1], self.cs_cam_angle[1] = self.update_angle(cam_euler[1], target_cam_euler[1], self.cs_cam_angle[1])
        cam_euler[2], self.cs_cam_angle[2] = self.update_angle(cam_euler[2], target_cam_euler[2], self.cs_cam_angle[2])
        cco = get_quaternion_from_euler(cam_euler)

        return cp,  cco

    def update_pos(self,c, e, cs):
        '''
        Desciption: Smooths the position animation
        Input:
            c: current position
            e: expected position
            cs: current speed
        Output:
            r: new position
        '''
        r = c
        # ending criterium
        if c == e:
            cs = 0
            return r,cs
        # prevent p control from making too small steps
        elif abs(c - e) <= (CHANGE_SPEED * 2):
            cs = 0
            r = e
        elif (c - e)> (CHANGE_SPEED * 2):
            # speed limiter if current value too close to end value
            brake_speed = CHANGE_SPEED * (c - e) * P_GAIN
            # accelerate as long as below speed limit
            if cs < brake_speed:
                cs += POS_ACC
            # adapt to speed limit
            else:
                cs = brake_speed
            r -= cs
        elif (e - c) > (CHANGE_SPEED * 2):
            brake_speed = CHANGE_SPEED * (e - c) * P_GAIN
            if cs < brake_speed:
                cs += POS_ACC
            else:
                cs = brake_speed
            r += cs
        return r, cs

    def cycle_distance(self,a, b):
        r = a - b
        # make sure angle is within [-math.pi, math.pi]
        if r > math.pi:
            r -= 2 * math.pi
        elif r < -math.pi:
            r += 2 * math.pi
        return r

    def update_angle(self,c, e, cs):
        r = c
        ce_dist= self.cycle_distance(c, e)
        ec_dist = self.cycle_distance(e, c)
        if c == e:
            cs = 0
            return r,cs
        elif abs(ce_dist) <= CHANGE_SPEED * 2:
            cs = 0
            r = e
        elif ce_dist > CHANGE_SPEED * 2:
            # speed limiter if current value too close to end value
            brake_speed = CHANGE_SPEED * self.cycle_distance(c, e) * P_GAIN
            # accelerate as long as below speed limit
            if cs < brake_speed:
                cs += ANGLE_ACC
            # adapt to speed limit
            else:
                cs = brake_speed
            r -= cs
        elif ec_dist > CHANGE_SPEED * 2:
            brake_speed = CHANGE_SPEED * ec_dist * P_GAIN
            if cs < brake_speed:
                cs += ANGLE_ACC
            else:
                cs = brake_speed
            r += cs
        # make sure angle is within [-math.pi, math.pi]
        if r > math.pi:
            r -= 2 * math.pi
        elif r < -math.pi:
            r += 2 * math.pi
        return r,cs
