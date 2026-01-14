import csv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d.axes3d import get_test_data

def filter(dt, updateNumber, **kwargs):

    if kwargs:
        print("Measurements available:")
        measurement_avail = True
        for key, value in kwargs.items():
            z = value
    else:
        measurement_avail = False
        print("No measurements available.")

    # initialize state
    if updateNumber == 1:
        # state vector
        filter.x = z #np.array([[0], [0], [0], [0], [0], [0]])
        # state convariance matrix
        filter.P = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        # state transition matrix
        filter.A = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        # state to measurement transition matrix
        filter.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        filter.HT = filter.H.transpose()
        # measurement uncertainty
        filter.R = 100 * np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        # noise convariance matrix
        filter.Q = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    # Predict State Forward
    x_p = filter.A.dot(filter.x)
    # Predict Covariance Forward
    P_p = filter.A.dot(filter.P).dot(filter.A.T) + filter.Q

    if measurement_avail == True:
        # Compute Kalman Gain
        S = filter.H.dot(P_p).dot(filter.HT) + filter.R
        K = np.dot(P_p.dot(filter.HT), np.linalg.inv(S))

        # Estimate State
        residual = z - np.squeeze(filter.H.dot(x_p))
        filter.x = np.squeeze(x_p) + np.dot(K, residual)
        # Estimate Covariance
        filter.P = P_p - K.dot(filter.H).dot(P_p)
    else:
        # no measurement update,
        # when measurements are not available
        filter.x = x_p
        filter.P = P_p

    return filter.x, filter.P

def rts_smoother(dt, updateNumber, x_kf, P_kf):

    # initialize state
    if updateNumber == 1:
        rts_smoother.x = x_kf
        rts_smoother.P = P_kf
        # state transition matrix
        rts_smoother.A = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        # noise convariance matrix
        rts_smoother.Q = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    else:
        # Predict State Forward
        x_p = rts_smoother.A.dot(x_kf)
        # Predict Covariance Forward
        P_p = rts_smoother.A.dot(P_kf).dot(rts_smoother.A.T) + rts_smoother.Q

        G = P_kf.dot(rts_smoother.A.T).dot(np.linalg.inv(P_p))
        x_residual = rts_smoother.x - x_p
        P_residual = rts_smoother.P - P_p

        # Estimate smoothed State
        rts_smoother.x = x_kf + np.dot(G, x_residual)
        # Estimate smoothed Covariance
        rts_smoother.P = P_kf + G.dot(P_residual).dot(G.T)

    return rts_smoother.x, rts_smoother.P

def extractSensorData(metadataFilePath):

    file = open(metadataFilePath, newline='')
    file_handler = csv.DictReader(file)
    if file_handler == None:
        print("Could not open drone file")
        exit(-1)

    time_msec = []
    latitude = []
    longitude = []
    altitude_m = []
    velocityX_mps = []
    velocityY_mps = []
    velocityZ_mps = []
    pitch_degree = []
    roll_degree = []
    yaw_degree = []
    gimbalPitch = []
    gimbalRoll = []
    gimbalYaw = []
    isTakingVideo = []

    i = 0

    for row in file_handler:
        if not row["MOV_Name"] or (row["MOV_Name"] != "BB36_statmob_sm1"):
            continue
        else:

            i = i + 1

            str_time_msec = float(row["time(millisecond)"])
            str_latitude = float(row["latitude"])
            str_longitude = float(row["longitude"])
            str_altitude_m = float(row["altitude(m)"])
            str_velocityX_mps = float(row["velocityX(mps)"])
            str_velocityY_mps = float(row["velocityY(mps)"])
            str_velocityZ_mps = float(row["velocityZ(mps)"])
            str_pitch_degree = float(row["pitch(deg)"])
            str_roll_degree = float(row["roll(deg)"])
            str_yaw_degree = float(row["yaw(deg)"])
            str_isTakingVideo = float(row["isTakingVideo"])
            str_gimbalPitch = float(row["pitchRaw"])
            str_gimbalRoll = float(row["rollRaw"])
            str_gimbalYaw = float(row["yawRaw"])

            time_msec.append(str_time_msec)
            latitude.append(str_latitude)
            longitude.append(str_longitude)
            altitude_m.append(str_altitude_m)
            velocityX_mps.append(str_velocityX_mps)
            velocityY_mps.append(str_velocityY_mps)
            velocityZ_mps.append(str_velocityZ_mps)
            pitch_degree.append(str_pitch_degree)
            roll_degree.append(str_roll_degree)
            yaw_degree.append(str_yaw_degree)
            isTakingVideo.append(str_isTakingVideo)
            gimbalPitch.append(str_gimbalPitch)
            gimbalRoll.append(str_gimbalRoll)
            gimbalYaw.append(str_gimbalYaw)

    sensorData = [time_msec, latitude, longitude, altitude_m, velocityX_mps, velocityY_mps, velocityZ_mps,
                      pitch_degree, roll_degree, yaw_degree, isTakingVideo, gimbalPitch, gimbalRoll, gimbalYaw]

    return sensorData

def getPose_IMU(metadataFilePath, dt):

    poseIMU = [[0, 0, 0]]
    droneVelocity = [[0, 0, 0]]

    yaw_camera = []
    pitch_camera = []
    roll_camera = []

    # pitch
    beta_prev = 0
    # roll
    gamma_prev = 0
    # yaw
    alpha_prev = 0

    Rx = np.array([[1, 0, 0], [0, np.cos(gamma_prev), -np.sin(gamma_prev)], [0, np.sin(gamma_prev), np.cos(gamma_prev)]])
    Ry = np.array([[np.cos(beta_prev), 0, np.sin(beta_prev)], [0, 1, 0], [-np.sin(beta_prev), 0, np.cos(beta_prev)]])
    Rz = np.array([[np.cos(alpha_prev), -np.sin(alpha_prev), 0], [np.sin(alpha_prev), np.cos(alpha_prev), 0], [0, 0, 1]])
    R_prev = Rz @ Ry @ Rx
    R_prev = R_prev

    for i in range(1, len(metadataFilePath[0])):

        # pitch
        pitch = math.radians(metadataFilePath[7][i])
        # roll
        roll = math.radians(metadataFilePath[8][i])
        # yaw
        yaw = math.radians(metadataFilePath[9][i])

        #Rx = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        #Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        #Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        #R = Rz @ Ry @ Rx

        r = Rot.from_rotvec([pitch, roll, -yaw])
        R = r.as_matrix()

        r_andOtherWay = Rot.from_rotvec([-pitch, -roll, yaw])
        R_andOtherWay = r_andOtherWay.as_matrix()

        R_motion = R_prev.T @ R
        r = r.from_matrix(R_motion)
        vec = r.as_rotvec()

        #yaw_c, pitch_c, roll_c = convertRotationMatrixToEulerAngle(R_motion)

        pitch_c = vec[0]
        roll_c = vec[1]
        yaw_c = vec[2]

        yaw_camera.append(yaw_c)
        pitch_camera.append(pitch_c)
        roll_camera.append(roll_c)

        R_prev = R

        velocity_t_mps = [-metadataFilePath[4][i], metadataFilePath[5][i], -metadataFilePath[6][i]]
        # convert velocity from mps (meter per second) to tps (meter per timestamp)
        scaleFactor = dt[i-1]
        velocity_t_tps = [t * scaleFactor for t in velocity_t_mps]

        poseIMU_current = np.array(poseIMU[-1]) + velocity_t_tps

        #poseIMU_current = R_motion @ poseIMU[-1] + velocity_t_tps

        poseIMU.append(poseIMU_current)
        droneVelocity.append(velocity_t_mps)

    return poseIMU, yaw_camera, pitch_camera, roll_camera, droneVelocity[1:]


def getPose_GPS(metadataFilePath):

    poseGPS = []
    start = True

    for i in range(len(metadataFilePath[0])):

        # convert to cartesian coordinates. coordinate origin in the middle of earth.
        latitude = math.radians(abs(metadataFilePath[1][i] - 90))
        longitude = math.radians(metadataFilePath[2][i])
        altitude = metadataFilePath[3][i] + 6371000

        x = altitude * np.sin(latitude) * np.cos(longitude)
        y = altitude * np.sin(latitude) * np.sin(longitude)
        z = altitude * np.cos(latitude)

        poseGPS_current = [x, y, z]

        # new origin start pose = [0, 0, 0]
        if start == True:
            theta = latitude
            phi = longitude
            start = False

        Rz = np.array([[np.cos(phi), np.sin(phi), 0],
                       [-np.sin(phi), np.cos(phi), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
                       [0, 1, 0],
                       [np.sin(theta), 0, np.cos(theta)]])

        poseGPS_current = Rz @ poseGPS_current
        poseGPS_current = Ry @ poseGPS_current
        poseGPS_current[2] = poseGPS_current[2] - 6371000
        poseGPS.append(poseGPS_current)

    return poseGPS


def convertRotationMatrixToEulerAngle(R):

    beta = np.arcsin(-R[2, 0])
    gamma = np.arcsin(R[2, 1] / np.cos(beta))
    alpha = np.arcsin(R[1, 0] / np.cos(beta))

    return np.array([alpha, beta, gamma])


if __name__ == '__main__':

    metadataFilePath = '/Users/piabideau/Development/datasets/predator-prey/Stationary_Mobile_Clips_for_Pia/corrected_flight_logs/_BB36_statmob_sm1.csv'

    frame_rate = 59.94005994005994  # frames per second
    duration_sec = 20  # total duration in seconds
    total_frames = int(frame_rate * duration_sec)
    # Generate timestamps for each frame
    t_frames = [(1.0 / frame_rate) * i for i in range(total_frames)]
    dt_frames = t_frames[1]

    sensorData = extractSensorData(metadataFilePath)
    t_sensor = (np.array(sensorData[0]) - sensorData[0][0]) / 1000 # in seconds
    dt_sensor = np.diff(t_sensor)
    # in meter
    dronePose, yaw_c, pitch_c, roll_c, droneVelocity = getPose_IMU(sensorData, dt_sensor)
    dronePoseGPS = getPose_GPS(sensorData)

    # ----------------
    # Kalman Filter
    # ----------------
    measPos = []
    measT = []
    estPos = []
    estVel = []
    estPosVar = []
    estVelVar = []
    posBound3Sigma = []
    for k in range(1, len(t_frames)):
        t = t_frames[k-1]
        measurement_idx = np.argmin(np.abs(t_sensor - t))
        t_idx = t_sensor[measurement_idx]
        outlier = (871043-sensorData[0][0])/1000
        if np.abs(t_idx - t) < dt_frames and t_idx != outlier:
            z = np.concatenate((dronePoseGPS[measurement_idx], droneVelocity[measurement_idx-1]))
            # Call Filter and return new State
            f, f_var = filter(dt_frames, k, measurement=z)
            measPos.append(z)
            measT.append(t)
        else:
            # Call Filter and return new State
            f, f_var = filter(dt_frames, k)
        # Save off that state so that it could be plotted
        estPos.append(f[0:3])
        estVel.append(f[3:6])
        estPosVar.append(np.array([f_var[0,0], f_var[1,1], f_var[2,2]]))
        estVelVar.append(np.array([f_var[3, 3], f_var[4, 4], f_var[5, 5]]))
        posVar = f_var
        posBound3Sigma.append(3 * np.sqrt(posVar[0][0]))

    # ----------------
    # Backward Filter
    # ----------------
    estPos_smooth = []
    estVel_smooth = []
    estPosVar_smooth = []
    estVelVar_smooth = []
    for k in range(len(t_frames)-1, 0, -1):
        x_kf = np.concatenate((estPos[k-1], estVel[k-1]))
        P_kf = np.eye(6) * np.concatenate((estPosVar[k-1], estVelVar[k-1]))
        f_smooth, f_smooth_var = rts_smoother(dt_frames, k-len(t_frames)+2, x_kf, P_kf)
        estPos_smooth.append(f_smooth[0:3])
        estVel_smooth.append(f_smooth[3:6])
        estPosVar_smooth.append(np.array([f_smooth_var[0, 0], f_smooth_var[1, 1], f_smooth_var[2, 2]]))
        estVelVar_smooth.append(np.array([f_smooth_var[3, 3], f_smooth_var[4, 4], f_smooth_var[5, 5]]))

    # ----------------
    # visualization
    # ----------------
    dronePose = np.asarray(measPos)
    t = np.asarray(measT)
    x = dronePose[:, 0]
    y = dronePose[:, 1]
    z = dronePose[:, 2]
    plt.figure('3D pose - measurement z')
    plt.plot(t[:], x, marker='o', markersize=3, label='x')
    plt.plot(t[:], y, marker='o', markersize=3, label='y')
    plt.plot(t[:], z, marker='o', markersize=3, label='height')
    plt.xlabel('time [in min]')
    plt.ylabel('distance [in meter]')
    plt.legend()

    dronePose = np.asarray(estPos)
    t = np.asarray(t_frames)
    x = dronePose[:, 0]
    y = dronePose[:, 1]
    z = dronePose[:, 2]
    dronePoseVar = np.sqrt(np.asarray(estPosVar))
    x_var = dronePoseVar[:, 0]
    y_var = dronePoseVar[:, 1]
    z_var = dronePoseVar[:, 2]
    plt.figure('3D pose - KF estimate')
    plt.plot(t[:-1], x, marker='o', markersize=3, label='x')
    plt.plot(t[:-1], y, marker='o', markersize=3, label='y')
    plt.plot(t[:-1], z, marker='o', markersize=3, label='height')
    plt.fill_between(t[:-1], x - x_var, x + x_var, alpha=.2)
    plt.fill_between(t[:-1], y - y_var, y + y_var, alpha=.2)
    plt.fill_between(t[:-1], z - z_var, z + z_var, alpha=.2)
    plt.xlabel('time [in min]')
    plt.ylabel('distance [in meter]')
    plt.legend()

    dronePose = np.asarray(estPos_smooth)
    t = np.asarray(t_frames)
    x = np.flip(dronePose[:, 0])
    y = np.flip(dronePose[:, 1])
    z = np.flip(dronePose[:, 2])
    dronePoseVar = np.sqrt(np.asarray(estPosVar_smooth))
    x_var = np.flip(dronePoseVar[:, 0])
    y_var = np.flip(dronePoseVar[:, 1])
    z_var = np.flip(dronePoseVar[:, 2])
    plt.figure('3D pose - KF-bf estimate ')
    plt.plot(t[:-1], x, marker='o', markersize=3, label='x')
    plt.plot(t[:-1], y, marker='o', markersize=3, label='y')
    plt.plot(t[:-1], z, marker='o', markersize=3, label='height')
    plt.fill_between(t[:-1], x - x_var, x + x_var, alpha=.2)
    plt.fill_between(t[:-1], y - y_var, y + y_var, alpha=.2)
    plt.fill_between(t[:-1], z - z_var, z + z_var, alpha=.2)
    plt.xlabel('time [in min]')
    plt.ylabel('distance [in meter]')
    plt.legend()

    plt.show()

'''
    fig = plt.figure('3D flight path - IMU')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=5)
    ax.plot3D(x, y, z)
    ax.set_xlabel("x [in meter]")
    ax.set_ylabel("y [in meter]")
    ax.set_zlabel("height [in meter]")
    ax.view_init(azim=0, elev=90)

    dronePoseGPS = np.asarray(dronePoseGPS)
    x = dronePoseGPS[0:M, 0]
    y = dronePoseGPS[0:M, 1]
    z = dronePoseGPS[0:M, 2]
    plt.figure('3D pose - GPS')
    plt.plot(t[0:M], x, marker='o', markersize=3, label='x')
    plt.plot(t[0:M], y, marker='o', markersize=3, label='y')
    plt.plot(t[0:M], z, marker='o', markersize=3,  label='height')
    plt.xlabel('time [in min]')
    plt.ylabel('distance [in meter]')
    plt.legend()

    fig = plt.figure('3D flight path - GPS')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[0:M], y[0:M], z[0:M], s=5)
    ax.plot3D(x[0:M], y[0:M], z[0:M])
    ax.set_xlabel("x [in meter]")
    ax.set_ylabel("y [in meter]")
    ax.set_zlabel("height [in meter]")
    ax.view_init(azim=0, elev=90)
    plt.legend()
'''