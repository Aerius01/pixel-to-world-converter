"""
Kalman filter implementations for camera pose estimation.

This module provides two Kalman filter variants for estimating camera position
and velocity from noisy GPS and IMU measurements:

1. StaticKalmanFilter: Uses constant noise matrices (current implementation)
2. AdaptiveKalmanFilter: Uses time-adaptive process noise (matches original C++)

Both implement the standard predict-update cycle for 6-DOF state estimation:
State vector: [x, y, z, vx, vy, vz] (position and velocity in meters and m/s)

The Kalman filter fuses GPS position measurements with IMU velocity measurements
to produce smooth, reliable camera pose estimates even when individual sensors
are noisy or temporarily unreliable.
"""

import numpy as np
import warnings

# Threshold for detecting ill-conditioned innovation covariance matrix
CONDITION_NUMBER_THRESHOLD = 1e10
# Small value added to diagonal when matrix is ill-conditioned
REGULARIZATION_EPSILON = 1e-6


class BaseKalmanFilter:
    """
    Base class for Kalman filter implementations.

    Implements the core predict-update cycle for linear Kalman filtering with
    constant velocity motion model. Subclasses must define noise matrices (Q, R, P).

    State Model:
        x_{k+1} = F * x_k + w_k   where w_k ~ N(0, Q)
        z_k = H * x_k + v_k       where v_k ~ N(0, R)

    Attributes:
        DOF: Degrees of freedom (state vector size, typically 6)
        state: Current state estimate [x, y, z, vx, vy, vz]
        P: State covariance matrix (DOF x DOF)
        Q: Process noise covariance (DOF x DOF)
        R: Measurement noise covariance (DOF x DOF)
        F: State transition matrix (updated each predict with dt)
        H: Measurement matrix (identity for direct state observation)
        I: Identity matrix (for numerical stability in updates)
    """
    def __init__(self, dof, initial_state):
        """
        Initialize base Kalman filter.

        Args:
            dof: Degrees of freedom (state vector size, typically 6)
            initial_state: Initial state estimate, ndarray of shape (dof,)
        """
        self.DOF = dof
        self.state = initial_state.copy()

        # State transition matrix F (constant velocity model, dt added during predict)
        self.F = np.eye(dof)
        # Measurement matrix H (identity = we observe full state directly)
        self.H = np.eye(dof)
        # Identity matrix for Joseph form covariance update
        self.I = np.eye(dof)

        # Noise matrices (set by subclasses)
        self.P = None  # State covariance
        self.Q = None  # Process noise covariance
        self.R = None  # Measurement noise covariance

    def predict(self, dt):
        """
        Predict next state using constant velocity motion model.

        Motion Model:
            x_{k+1} = x_k + vx_k * dt
            y_{k+1} = y_k + vy_k * dt
            z_{k+1} = z_k + vz_k * dt
            vx_{k+1} = vx_k  (constant velocity)
            vy_{k+1} = vy_k
            vz_{k+1} = vz_k

        Updates F matrix with time step dt, then propagates state and covariance:
            state = F * state
            P = F * P * F^T + Q

        Args:
            dt: Time step in seconds since last predict/update
        """
        # Update state transition matrix with dt (position = position + velocity * dt)
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

        # Allow subclasses to update process noise based on dt (e.g., AdaptiveKalmanFilter)
        self._update_process_noise(dt)

        # Propagate state: x = F * x
        self.state = self.F @ self.state

        # Propagate covariance: P = F * P * F^T + Q
        temp = self.F @ self.P
        self.P = temp @ self.F.T + self.Q

    def _update_process_noise(self, dt):
        """
        Update process noise matrix Q based on time step.

        Base implementation does nothing (static Q). Subclasses like AdaptiveKalmanFilter
        override this to compute time-adaptive noise.

        Args:
            dt: Time step in seconds
        """
        pass

    def update(self, z):
        """
        Update state estimate with new measurement using Kalman gain.

        Measurement Update (Kalman gain formulation):
            y = z - H * x          (innovation/residual)
            S = H * P * H^T + R    (innovation covariance)
            K = P * H^T * S^{-1}   (Kalman gain)
            x = x + K * y          (state update)
            P = (I - K*H) * P * (I - K*H)^T + K * R * K^T  (Joseph form)

        The Joseph form covariance update is used for numerical stability. It ensures
        P remains positive semi-definite even with rounding errors.

        If innovation covariance S is ill-conditioned (cond > 1e10), adds small
        regularization term to diagonal for numerical stability.

        Args:
            z: Measurement vector, ndarray of shape (DOF,)
        """
        # Compute innovation (residual): y = z - H*x
        y = z - (self.H @ self.state)

        # Compute innovation covariance: S = H*P*H^T + R
        temp = self.H @ self.P
        S = temp @ self.H.T + self.R

        # Check for ill-conditioned S matrix (numerical stability)
        cond = np.linalg.cond(S)
        if cond > CONDITION_NUMBER_THRESHOLD:
            warnings.warn(
                f"Ill-conditioned innovation covariance (cond={cond:.2e}). "
                f"Adding regularization term.",
                RuntimeWarning
            )
            S += np.eye(len(S)) * REGULARIZATION_EPSILON

        # Compute Kalman gain: K = P*H^T*S^{-1}
        # Use solve instead of inv for numerical stability
        PHt = self.P @ self.H.T
        K = np.linalg.solve(S.T, PHt.T).T

        # Update state: x = x + K*y
        self.state = self.state + (K @ y)

        # Update covariance using Joseph form: P = (I-K*H)*P*(I-K*H)^T + K*R*K^T
        # This form is more numerically stable than the simple P = (I-K*H)*P
        IKH = self.I - (K @ self.H)
        temp1 = IKH @ self.P
        temp2 = K @ self.R
        self.P = (temp1 @ IKH.T) + (temp2 @ K.T)

        # Enforce symmetry (covariance must be symmetric due to rounding errors)
        self.P = (self.P + self.P.T) / 2


class StaticKalmanFilter(BaseKalmanFilter):
    """
    Kalman filter with constant (static) noise matrices.

    This is the current implementation used in the pipeline. Noise matrices Q, R, and P
    are set once during initialization and remain constant throughout filtering.

    This differs from the original C++ implementation which used time-adaptive process
    noise (see AdaptiveKalmanFilter). The static matrices date back to Pia's port.

    Typical Usage:
        kf = StaticKalmanFilter(
            initial_state=np.concatenate([gps_pos, imu_vel]),
            R=100 * np.eye(6),  # Measurement noise
            Q=1 * np.eye(6),    # Process noise
            P_initial=1 * np.eye(6)  # Initial uncertainty
        )
        for each frame:
            kf.predict(dt)
            kf.update(measurement)
    """
    def __init__(self, initial_state, R, Q, P_initial):
        """
        Initialize static Kalman filter with constant noise matrices.

        Args:
            initial_state: Initial state estimate [x, y, z, vx, vy, vz]
            R: Measurement noise covariance matrix (6x6)
            Q: Process noise covariance matrix (6x6)
            P_initial: Initial state covariance matrix (6x6)
        """
        dof = len(initial_state)
        super().__init__(dof, initial_state)

        self.R = R.copy()
        self.Q = Q.copy()
        self.P = P_initial.copy()


class AdaptiveKalmanFilter(BaseKalmanFilter):
    """
    Kalman filter with time-adaptive process noise.

    This implementation matches the original C++ codebase (master_thesis_project-main-HU).
    Process noise Q is recomputed at each time step based on dt, allowing the filter
    to adapt to varying time intervals between measurements.

    Currently NOT used in the pipeline (StaticKalmanFilter is used instead). This
    class is preserved for potential future investigation of optimal noise tuning.

    Adaptive Process Noise Model:
        Q[i,i] = out_q * in_q^2 * dt^4  for position components (i < 3)
        Q[i,i] = out_q * dt^2           for velocity components (i >= 3)

    This model assumes position uncertainty grows as dt^4 (integrated twice from
    acceleration noise), while velocity uncertainty grows as dt^2.
    """
    def __init__(self, initial_state, pos_variance, alt_variance, vel_variance,
                 in_q, out_q, P_initial_variance):
        """
        Initialize adaptive Kalman filter with time-dependent process noise.

        Args:
            initial_state: Initial state estimate [x, y, z, vx, vy, vz]
            pos_variance: Measurement noise variance for horizontal position (x, y)
            alt_variance: Measurement noise variance for altitude (z)
            vel_variance: Measurement noise variance for velocity (vx, vy, vz)
            in_q: Inner process noise factor (acceleration noise magnitude)
            out_q: Outer process noise scaling factor
            P_initial_variance: Initial state covariance diagonal value

        Raises:
            ValueError: If any variance or noise parameter is non-positive
        """
        dof = len(initial_state)
        super().__init__(dof, initial_state)

        if pos_variance <= 0 or alt_variance <= 0 or vel_variance <= 0 or out_q <= 0 or in_q <= 0:
            raise ValueError("All variance and noise parameters must be positive")

        # Construct measurement noise matrix R with per-state variances
        self.R = np.eye(dof)
        for i in range(dof):
            if i < 2:  # Horizontal position (x, y)
                self.R[i, i] = pos_variance
            elif i == 2:  # Altitude (z)
                self.R[i, i] = alt_variance
            else:  # Velocity components (vx, vy, vz)
                self.R[i, i] = vel_variance

        # Store noise scaling factors for adaptive Q computation
        self.in_q = in_q
        self.out_q = out_q

        # Process noise Q will be computed adaptively in _update_process_noise
        self.Q = np.zeros((dof, dof))

        # Initialize state covariance as scaled identity
        self.P = np.eye(dof) * P_initial_variance

    def _update_process_noise(self, dt):
        """
        Update process noise matrix Q based on time step (time-adaptive).

        Implements the original C++ adaptive process noise model:
            Q[i,i] = out_q * in_q^2 * dt^4  for position (i < 3)
            Q[i,i] = out_q * dt^2           for velocity (i >= 3)

        This models uncertainty growth from continuous-time acceleration noise,
        integrated to position (2 integrations) and velocity (1 integration).

        Args:
            dt: Time step in seconds
        """
        # Precompute powers of dt for efficiency
        q2 = self.in_q ** 2
        dt2 = dt * dt
        dt3 = dt * dt2
        dt4 = dt * dt3

        # Reset Q to zero (diagonal only will be filled)
        self.Q.fill(0)

        # Set diagonal elements based on state type
        for i in range(self.DOF):
            if i < 3:  # Position components: uncertainty ~ dt^4
                self.Q[i, i] = q2 * dt4
            else:  # Velocity components: uncertainty ~ dt^2
                self.Q[i, i] = dt2

        # Scale by outer factor
        self.Q = self.Q * self.out_q
