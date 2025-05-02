import torch
import numpy as np

def dubins_dynamics_tensor(state, action, dt):
    """
    Dubins car dynamics with velocity as a state variable.
    
    Args:
        state: torch.Tensor of shape (batch_size, 4) containing [x, y, theta, v]
        action: torch.Tensor of shape (batch_size, 2) containing [angular_velocity, linear_acceleration]
        dt: float, time step
        
    Returns:
        next_state: torch.Tensor of shape (batch_size, 4) containing the next state
    """
    # Extract state components
    x = state[..., 0]
    y = state[..., 1]
    theta = state[..., 2]
    v = state[..., 3]
    
    # Extract action components
    angular_velocity = action[..., 0]
    linear_acceleration = action[..., 1]
    
    # Compute next state
    next_x = x + v * torch.sin(theta) * dt
    next_y = y + v * torch.cos(theta) * dt
    next_theta = theta + angular_velocity * dt
    next_v = v + linear_acceleration * dt
    
    # Normalize theta to [0, 2*pi)
    next_theta = next_theta % (2 * np.pi)
    
    # Combine into next state
    next_state = torch.stack([next_x, next_y, next_theta, next_v], dim=-1)
    
    return next_state