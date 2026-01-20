"""Kalman filter implementations for camera pose estimation."""

import numpy as np
import warnings

# Numerical stability constants
CONDITION_NUMBER_THRESHOLD = 1e10  # Warn if matrix is ill-conditioned
REGULARIZATION_EPSILON = 1e-6      # Regularization term for ill-conditioned matrices


class BaseKalmanFilter:
    """
    Base class for Kalman filter implementations.

    Provides common predict() and update() methods for all Kalman filter variants.
    Subclasses must implement their own __init__() to configure noise matrices.
    """

    def __init__(self, dof, initial_state):
        """
        Initialize base Kalman filter structure.

        Args:
            dof: Degrees of freedom (state dimension)
            initial_state: Initial state vector
        """
        self.DOF = dof
        self.state = initial_state.copy()

        # State transition and observation matrices (identity by default)
        self.F = np.eye(dof)
        self.H = np.eye(dof)
        self.I = np.eye(dof)

        # Matrices to be set by subclasses: P, Q, R
        self.P = None  # State covariance
        self.Q = None  # Process noise covariance
        self.R = None  # Measurement noise covariance

    def predict(self, dt):
        """
        Kalman filter prediction step.

        Args:
            dt: Time step in seconds
        """
        # Update state transition matrix with time step
        # Assumes state vector: [x, y, z, vx, vy, vz]
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # This method will be overridden by subclasses if they need custom Q computation
        self._update_process_noise(dt)

        # Standard prediction equations
        # Optimized: use @ operator and minimize intermediate allocations
        self.state = self.F @ self.state
        # Compute F*P*F^T + Q efficiently by reusing intermediate result
        temp = self.F @ self.P
        self.P = temp @ self.F.T + self.Q

    def _update_process_noise(self, dt):
        """
        Update process noise matrix Q. Override in subclasses for custom behavior.

        Args:
            dt: Time step in seconds
        """
        pass  # Default: Q remains constant

    def update(self, z):
        """
        Kalman filter update (measurement) step.

        Args:
            z: Measurement vector
        """
        # Compute innovation (residual)
        y = z - (self.H @ self.state)

        # Compute innovation covariance S = H*P*H^T + R
        # Reuse intermediate result to avoid extra allocation
        temp = self.H @ self.P
        S = temp @ self.H.T + self.R

        # Check condition number for numerical stability
        cond = np.linalg.cond(S)
        if cond > CONDITION_NUMBER_THRESHOLD:
            warnings.warn(
                f"Ill-conditioned innovation covariance (cond={cond:.2e}). "
                f"Adding regularization term.",
                RuntimeWarning
            )
            # Add regularization to diagonal
            S += np.eye(len(S)) * REGULARIZATION_EPSILON

        # Compute Kalman gain using solve (more stable than explicit inverse)
        # K = P*H^T*inv(S) = solve(S^T, (P*H^T)^T)^T
        PHt = self.P @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T

        # Update state vector
        self.state = self.state + (K @ y)

        # Update covariance matrix using Joseph form (numerically stable)
        IKH = self.I - (K @ self.H)
        temp1 = IKH @ self.P
        temp2 = K @ self.R
        self.P = (temp1 @ IKH.T) + (temp2 @ K.T)

        # Ensure covariance remains symmetric (numerical errors can break symmetry)
        self.P = (self.P + self.P.T) / 2


class StaticKalmanFilter(BaseKalmanFilter):
    """
    Kalman filter with constant noise matrices.

    All noise matrices (R, Q, P) are fixed at initialization and do not change
    during operation. This is the most common use case for well-tuned systems.
    """

    def __init__(self, initial_state, R, Q, P_initial):
        """
        Initialize Kalman filter with static noise matrices.

        Args:
            initial_state: Initial state vector (numpy array of shape [dof])
            R: Measurement noise covariance matrix (numpy array of shape [dof, dof])
            Q: Process noise covariance matrix (numpy array of shape [dof, dof])
            P_initial: Initial state covariance matrix (numpy array of shape [dof, dof])
        """
        dof = len(initial_state)
        super().__init__(dof, initial_state)

        self.R = R.copy()
        self.Q = Q.copy()
        self.P = P_initial.copy()


class AdaptiveKalmanFilter(BaseKalmanFilter):
    """
    Kalman filter with time-adaptive process noise.

    The process noise matrix Q is computed dynamically at each prediction step
    based on the time interval dt and scaling factors (in_q, out_q). This matches
    the original C++ implementation behavior.
    """

    def __init__(self, initial_state, pos_variance, alt_variance, vel_variance,
                 in_q, out_q, P_initial_variance):
        """
        Initialize Kalman filter with adaptive process noise.

        Args:
            initial_state: Initial state vector (numpy array of shape [dof])
            pos_variance: Measurement noise for x,y positions
            alt_variance: Measurement noise for z position
            vel_variance: Measurement noise for velocities
            in_q: Inner process noise factor (time-adaptive scaling)
            out_q: Outer process noise factor (overall Q scaling)
            P_initial_variance: Scalar variance for initial covariance (P = P_initial_variance * I)
        """
        dof = len(initial_state)
        super().__init__(dof, initial_state)

        # Validate parameters
        if pos_variance <= 0 or alt_variance <= 0 or vel_variance <= 0 or out_q <= 0 or in_q <= 0:
            raise ValueError("All variance and noise parameters must be positive")

        # Configure measurement noise matrix R
        self.R = np.eye(dof)
        for i in range(dof):
            if i < 2:
                self.R[i, i] = pos_variance
            elif i == 2:
                self.R[i, i] = alt_variance
            else:
                self.R[i, i] = vel_variance

        # Store noise factors for dynamic Q computation
        self.in_q = in_q
        self.out_q = out_q

        # Initialize process noise matrix (will be recomputed in predict)
        self.Q = np.zeros((dof, dof))

        # Initialize state covariance
        self.P = np.eye(dof) * P_initial_variance

    def _update_process_noise(self, dt):
        """
        Compute time-adaptive process noise matrix Q.

        The Q matrix is scaled by dt^2 and dt^4 terms to account for acceleration
        uncertainty that accumulates over time.

        Args:
            dt: Time step in seconds
        """
        q2 = self.in_q ** 2
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3

        self.Q.fill(0)
        for i in range(self.DOF):
            if i < 3:  # Position states
                self.Q[i, i] = q2 * dt4
            else:  # Velocity states
                self.Q[i, i] = dt2

        self.Q = self.Q * self.out_q
