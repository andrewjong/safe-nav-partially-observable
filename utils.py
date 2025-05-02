import torch

# Text color constants
BLACK   = "\033[30m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"
RESET   = "\033[0m"

# Bold text color constants
BOLD_BLACK   = "\033[1;30m"
BOLD_RED     = "\033[1;31m"
BOLD_GREEN   = "\033[1;32m"
BOLD_YELLOW  = "\033[1;33m"
BOLD_BLUE    = "\033[1;34m"
BOLD_MAGENTA = "\033[1;35m"
BOLD_CYAN    = "\033[1;36m"
BOLD_WHITE   = "\033[1;37m"


def dubins_dynamics_tensor(
    current_state: torch.Tensor, action: torch.Tensor, dt: float
) -> torch.Tensor:
    """
    current_state: shape(num_samples, dim_x)
    action: shape(num_samples, dim_u)
    
    action[:, 0] is angular velocity
    action[:, 1] is linear acceleration
    Implemented discrete time dynamics with RK-4.
    return:
    next_state: shape(num_samples, dim_x)
    """
    def one_step_dynamics(state, action):
        """Compute the derivatives [dx/dt, dy/dt, dtheta/dt, dv/dt]."""
        # Extract state variables
        x, y, theta, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        angular_vel = action[:, 0]
        linear_acc = action[:, 1]
        
        # Compute derivatives
        dx_dt = v * torch.cos(theta)
        dy_dt = v * torch.sin(theta)
        dtheta_dt = angular_vel
        dv_dt = linear_acc
        
        # Stack derivatives into a tensor with the same shape as state
        derivatives = torch.stack([dx_dt, dy_dt, dtheta_dt, dv_dt], dim=1)
        return derivatives
    
    # k1
    k1 = one_step_dynamics(current_state, action)
    # k2
    mid_state_k2 = current_state + 0.5 * dt * k1
    k2 = one_step_dynamics(mid_state_k2, action)
    # k3
    mid_state_k3 = current_state + 0.5 * dt * k2
    k3 = one_step_dynamics(mid_state_k3, action)
    # k4
    end_state_k4 = current_state + dt * k3
    k4 = one_step_dynamics(end_state_k4, action)
    # Combine k1, k2, k3, k4 to compute the next state
    next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Normalize theta to [-pi, pi]
    # normalize theta to [0, 2*pi]
    next_state[..., 2] = next_state[..., 2] % (2 * torch.pi)
    return next_state
