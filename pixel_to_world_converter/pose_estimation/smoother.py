"""
Rauch-Tung-Striebel (RTS) smoother for backward-pass state refinement.

The RTS smoother performs a backward pass through Kalman-filtered data to produce
refined state estimates that incorporate "future knowledge". While the Kalman
filter is causal (only uses past measurements), the smoother is acausal and can
leverage the entire dataset to improve estimates at each time step.

Mathematical Concept:
    Forward Kalman filter: x_k|k = f(z_1, ..., z_k)     [causal]
    RTS smoother: x_k|N = f(z_1, ..., z_N)              [acausal, N = last frame]

The smoothed estimates have lower variance than the filtered estimates because
they incorporate information from both past and future measurements.
"""

import numpy as np


class RTSSmoother:
    """
    Rauch-Tung-Striebel smoother for backward-pass state refinement.

    Processes Kalman filter estimates in reverse chronological order to produce
    smoothed state estimates. Each iteration refines the current time step by
    incorporating information from future time steps via the smoother gain G.

    Algorithm (for each k from N-1 down to 0):
        1. Predict forward: x_p = A * x_kf (what Kalman predicted for k+1)
        2. Compute smoother gain: G = P_kf * A^T * P_p^{-1}
        3. Smooth state: x_s = x_kf + G * (x_{k+1|N} - x_p)
        4. Smooth covariance: P_s = P_kf + G * (P_{k+1|N} - P_p) * G^T

    Attributes:
        dt: Time step in seconds (constant for all frames)
        dof: Degrees of freedom (state vector size, typically 6)
        initialized: Whether smoother has been initialized with last state
        x: Current smoothed state estimate
        P: Current smoothed covariance estimate
        A: State transition matrix (constant velocity model)
        Q: Process noise covariance (identity for this implementation)
    """
    def __init__(self, dt, dof=6):
        """
        Initialize RTS smoother.

        Args:
            dt: Time step in seconds (assumed constant for all frames)
            dof: Degrees of freedom (state vector size, default 6)
        """
        self.dt = dt
        self.dof = dof
        self.initialized = False

        self.x = None  # Current smoothed state
        self.P = None  # Current smoothed covariance

        # State transition matrix A (constant velocity model)
        self.A = np.eye(dof)

        # For 6-DOF state [x, y, z, vx, vy, vz], add velocity->position coupling
        if dof >= 6:
            for i in range(min(3, dof // 2)):
                self.A[i, i + dof // 2] = dt  # position += velocity * dt

        # Process noise Q (identity for this implementation)
        self.Q = np.eye(dof)

    def update(self, x_kf, P_kf):
        """
        Perform one RTS smoother update step (backward in time).

        Must be called in reverse chronological order (from last frame to first).
        The first call (last frame) simply stores the Kalman estimate. Subsequent
        calls refine earlier estimates using information from later frames.

        Algorithm:
            if not initialized:
                x_s = x_kf, P_s = P_kf  (last frame = Kalman estimate)
            else:
                x_p = A * x_kf                    (predict what Kalman thought)
                P_p = A * P_kf * A^T + Q          (predicted covariance)
                G = P_kf * A^T * P_p^{-1}         (smoother gain)
                x_s = x_kf + G * (x_{next} - x_p) (incorporate future knowledge)
                P_s = P_kf + G * (P_{next} - P_p) * G^T

        Args:
            x_kf: Kalman filtered state at current time step, ndarray of shape (dof,)
            P_kf: Kalman filtered covariance at current time step, ndarray of shape (dof, dof)

        Returns:
            tuple: (x_s, P_s)
                - x_s: Smoothed state estimate
                - P_s: Smoothed covariance estimate
        """
        # First call (last frame): initialize with Kalman estimate
        if not self.initialized:
            self.x = x_kf
            self.P = P_kf
            self.initialized = True
            return self.x, self.P

        # Predict what Kalman filter would have estimated for next frame
        x_p = self.A @ x_kf
        temp = self.A @ P_kf
        P_p = temp @ self.A.T + self.Q

        # Compute smoother gain G = P_kf * A^T * P_p^{-1}
        # Use solve instead of inv for numerical stability
        P_kf_AT = P_kf @ self.A.T
        G = np.linalg.solve(P_p.T, P_kf_AT.T).T

        # Compute residuals (difference between smoothed next state and predicted)
        x_residual = self.x - x_p
        P_residual = self.P - P_p

        # Update smoothed state: x_s = x_kf + G * (x_{next} - x_p)
        self.x = x_kf + (G @ x_residual)

        # Update smoothed covariance: P_s = P_kf + G * (P_{next} - P_p) * G^T
        temp_P = G @ P_residual
        self.P = P_kf + (temp_P @ G.T)

        return self.x, self.P

    def reset(self):
        """
        Reset smoother state for processing a new trajectory.

        Clears internal state so the smoother can be reused.
        """
        self.initialized = False
        self.x = None
        self.P = None
