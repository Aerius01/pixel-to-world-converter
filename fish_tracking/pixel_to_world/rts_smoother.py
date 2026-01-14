"""
Rauch-Tung-Striebel (RTS) smoother for backward pass smoothing.

The RTS smoother processes Kalman filter estimates in reverse chronological order
to produce smoothed estimates that incorporate both past and future measurements.
"""

import numpy as np


class RTSSmoother:
    """
    Rauch-Tung-Striebel (RTS) smoother for backward pass smoothing.
    
    Processes Kalman filter estimates in reverse to produce smoothed estimates
    using both past and future measurements. This improves state estimation
    by incorporating information from the entire time series.
    """
    
    def __init__(self, dt, dof=6):
        """
        Initialize the RTS smoother.
        
        Args:
            dt (float): Time step in seconds
            dof (int): Degrees of freedom (state dimension), default 6 for [x, y, z, vx, vy, vz]
        """
        self.dt = dt
        self.dof = dof
        self.initialized = False
        
        # Smoothed state and covariance
        self.x = None
        self.P = None
        
        # State transition matrix (constant velocity model for 6-DOF)
        # Can be extended for other DOF configurations
        if dof == 6:
            self.A = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
        else:
            # Generic identity matrix for other DOF values
            self.A = np.eye(dof)
            # Add velocity integration for position states if dof allows
            if dof >= 6:
                for i in range(min(3, dof // 2)):
                    self.A[i, i + dof // 2] = dt
        
        # Process noise covariance matrix
        self.Q = np.eye(dof)
    
    def update(self, x_kf, P_kf):
        """
        Update the smoother with Kalman filter estimates.
        
        This should be called in reverse chronological order (from end to start).
        
        Args:
            x_kf (np.array): Kalman filter state estimate
            P_kf (np.array): Kalman filter covariance estimate
        
        Returns:
            tuple: (smoothed_state, smoothed_covariance)
        """
        # Initialize on first update (which is the last time step when going backward)
        if not self.initialized:
            self.x = x_kf
            self.P = P_kf
            self.initialized = True
            return self.x, self.P
        
        # Predict forward from current KF estimate
        x_p = self.A.dot(x_kf)
        P_p = self.A.dot(P_kf).dot(self.A.T) + self.Q
        
        # Compute smoother gain
        G = P_kf.dot(self.A.T).dot(np.linalg.inv(P_p))
        
        # Compute residuals between smoothed and predicted
        x_residual = self.x - x_p
        P_residual = self.P - P_p
        
        # Compute smoothed estimates
        self.x = x_kf + np.dot(G, x_residual)
        self.P = P_kf + G.dot(P_residual).dot(G.T)
        
        return self.x, self.P
    
    def reset(self):
        """Reset the smoother to uninitialized state."""
        self.initialized = False
        self.x = None
        self.P = None
