import numpy as np

class KalmanFilter:
    def __init__(self,KF_DOF,INIT_STATE_VAR):
        self.state = np.zeros(KF_DOF)
        self.DOF = KF_DOF
        self.P = np.eye(KF_DOF) * INIT_STATE_VAR
        self.F = np.eye(KF_DOF)
        self.H = np.eye(KF_DOF)
        self.R = np.eye(KF_DOF)
        self.Q = np.zeros((KF_DOF, KF_DOF))
        self.I = np.eye(KF_DOF)
        self.out_q = 0.0
        self.in_q = 0.0


    def init(self, pos_variance, alt_variance, vel_variance, in_q, out_q, initial_state, 
             R_override=None, Q_override=None, P_initial_override=None):
        """
        Initialize the Kalman filter.
        
        Args:
            pos_variance: Measurement noise for x,y positions (used if R_override is None)
            alt_variance: Measurement noise for z position (used if R_override is None)
            vel_variance: Measurement noise for velocities (used if R_override is None)
            in_q: Inner process noise factor (used if Q_override is None)
            out_q: Outer process noise factor (used if Q_override is None)
            initial_state: Initial state vector
            R_override: Optional custom measurement noise matrix. If provided, overrides 
                       the computed R matrix from pos/alt/vel variances.
            Q_override: Optional custom process noise matrix. If provided, overrides 
                       the time-adaptive Q computation in predict().
            P_initial_override: Optional custom initial covariance matrix. If provided, overrides
                              the default P = INIT_STATE_VAR * I initialization.
        """
        if pos_variance <= 0 or alt_variance <= 0 or vel_variance <= 0 or out_q <= 0 or in_q <= 0:
            print("INVALID_INITIALIZATION_VALUES")
            exit(-1)

        # Configure measurement noise (R matrix)
        if R_override is not None:
            self.R = R_override.copy()
        else:
            # Standard configuration: different variances for different state types
            for i in range(self.DOF):
                if i < 2:
                    self.R[i, i] = pos_variance
                elif i == 2:
                    self.R[i, i] = alt_variance
                else:
                    self.R[i, i] = vel_variance

        self.in_q = in_q
        self.out_q = out_q
        
        # Configure process noise (Q matrix)
        if Q_override is not None:
            self.Q = Q_override.copy()
            self.Q_override_active = True
        else:
            self.Q_override_active = False

        # Configure initial covariance (P matrix)
        if P_initial_override is not None:
            self.P = P_initial_override.copy()

        self.state = initial_state

    def predict(self, dt):
        # Set self.F according to time step
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Compute process noise Q (skip if Q_override was provided)
        if not hasattr(self, 'Q_override_active') or not self.Q_override_active:
            q2 = self.in_q**2
            dt2 = dt * dt
            dt3 = dt * dt2
            dt4 = dt * dt3

            self.Q.fill(0)
            for i in range(self.DOF):
                if i < 3:
                    self.Q[i, i] = q2 * dt4
                else:
                    self.Q[i, i] = dt2

            self.Q = self.Q * self.out_q
        # else: use the constant Q matrix set during init
        #print("Q:",self.Q)

        # Apply prediction step
        self.state = np.dot(self.F, self.state)
        # Update covariance matrix
        Ft = self.F.transpose()
        self.P = np.dot(np.dot(self.F, self.P), Ft) + self.Q
        #print("P:",self.P)


    def update(self, z):
        Ht = self.H.transpose()

        PHt = np.dot(self.P, Ht)
        y = z - np.dot(self.H, self.state)
        S = np.dot(self.H, PHt) + self.R

        # changed inverse to pinv because of singular matrix
        K = np.dot(PHt, np.linalg.inv(S))

        # Update state vector
        self.state = self.state + np.dot(K, y)
        
        # Update covariance matrix using Joseph form (numerically stable)
        IKH = self.I - np.dot(K, self.H)
        self.P = np.dot(np.dot(IKH, self.P), IKH.transpose()) + np.dot(np.dot(K, self.R), K.transpose())
        #print("state after:",self.state)
