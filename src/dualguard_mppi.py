import numpy as np
import torch
from src.mppi import MPPI, Navigator
from utils import dubins_dynamics_tensor

class DualGuardMPPI(MPPI):
    """
    DualGuard MPPI: An extension of MPPI that uses HJ Reachability to ensure safety during sampling.
    
    This implementation:
    1. Uses the gradient of HJ values to guide sampling away from unsafe regions
    2. Checks the final trajectory against the unsafe set to ensure safety
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        hj_solver=None,  # HJ reachability solver
        safety_threshold=0.0,  # Value above which states are considered safe
        gradient_scale=1.0,  # Scale factor for gradient-based corrections
        **kwargs
    ):
        super().__init__(dynamics, running_cost, nx, noise_sigma, **kwargs)
        self.hj_solver = hj_solver
        self.safety_threshold = safety_threshold
        self.gradient_scale = gradient_scale
        self.hj_values = None
        self.hj_values_grad = None
        
    def set_hj_values(self, values, values_grad=None):
        """Set the HJ reachability values and gradients."""
        self.hj_values = values
        if values_grad is None and values is not None:
            # Compute gradients if not provided
            self.hj_values_grad = np.gradient(values)
        else:
            self.hj_values_grad = values_grad
    
    def _compute_total_cost_batch(self):
        """
        Override the sampling process to incorporate HJ reachability guidance.
        """
        # Sample noise as in the original MPPI
        self.noise = self.noise_dist.sample((self.K, self.T))
        
        # Apply HJ reachability guidance to the noise if values are available
        if self.hj_values is not None and self.hj_solver is not None:
            self._apply_hj_guidance_to_noise()
        
        # Compute perturbed actions
        self.perturbed_action = self.U + self.noise
        
        # Handle null action sampling
        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0
            
        # Bound the actions
        self.perturbed_action = self._bound_action(self.perturbed_action)
        
        # Update noise after bounding
        self.noise = self.perturbed_action - self.U
        
        # Compute action cost
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv
        
        # Compute rollout costs
        self.cost_total, self.states, self.actions = self._compute_rollout_costs(
            self.perturbed_action
        )
        self.actions /= self.u_scale
        
        # Add action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total += perturbation_cost
        
        # Apply safety check to the rollouts
        if self.hj_values is not None and self.hj_solver is not None:
            self._apply_safety_cost_to_rollouts()
            
        return self.cost_total
    
    def _apply_hj_guidance_to_noise(self):
        """
        Apply HJ reachability guidance to the noise samples.
        This steers the sampling away from unsafe regions.
        """
        # Current state
        state = self.state.cpu().numpy()
        
        # Get state indices in the grid
        state_ind = self.hj_solver.state_to_grid(state)
        
        # Extract the gradient at the current state
        grad_x = self.hj_values_grad[0][tuple(state_ind)]
        grad_y = self.hj_values_grad[1][tuple(state_ind)]
        grad_theta = self.hj_values_grad[2][tuple(state_ind)]
        grad_v = self.hj_values_grad[3][tuple(state_ind)]
        
        # Gradient of the value function at the current state
        grad_V = np.array([grad_x, grad_y, grad_theta, grad_v])
        
        # Extract state components
        x, y, theta, v = state
        
        # Open loop dynamics contribution
        f_x = np.array([v * np.sin(theta), v * np.cos(theta), 0.0, 0.0])
        
        # Control jacobian
        g_x = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        
        # Compute the direction that increases safety
        lhs = grad_V @ g_x  # Left-hand side coefficient for u
        
        # If the gradient is significant, use it to guide the noise
        if np.linalg.norm(lhs) > 1e-6:
            # Convert to torch tensor
            lhs_tensor = torch.tensor(lhs, device=self.d, dtype=self.dtype)
            
            # Scale the noise based on the gradient direction
            # This adds a bias to the noise that pushes it toward safer regions
            for t in range(self.T):
                # Scale the guidance based on time step (stronger guidance for earlier steps)
                time_scale = 1.0 / (1.0 + 0.1 * t)
                
                # Apply the guidance to the noise
                safety_bias = time_scale * self.gradient_scale * lhs_tensor
                self.noise[:, t] += safety_bias
    
    def _apply_safety_cost_to_rollouts(self):
        """
        Apply an additional safety cost to the rollouts based on HJ values.
        This penalizes trajectories that enter unsafe regions.
        """
        # Get the states from the rollouts
        # states shape: M x K x T x nx
        M, K, T, nx = self.states.shape
        
        # Initialize safety cost
        safety_cost = torch.zeros((M, K), device=self.d, dtype=self.dtype)
        
        # Check each state in the trajectory for safety
        for m in range(M):
            for k in range(K):
                for t in range(T):
                    # Get the state
                    state = self.states[m, k, t].cpu().numpy()
                    
                    # Check if the state is safe
                    is_safe, value, _ = self.hj_solver.check_if_safe(state, self.hj_values)
                    
                    # Add a large cost for unsafe states
                    if not is_safe:
                        # Penalize more for earlier violations
                        time_penalty = 10.0 / (1.0 + 0.1 * t)
                        safety_cost[m, k] += time_penalty * 100.0
        
        # Add the safety cost to the total cost
        self.cost_total += safety_cost.mean(dim=0)
    
    def command(self, state):
        """
        Override the command method to ensure the final trajectory is safe.
        
        Args:
            state: Current state of the system
            
        Returns:
            action: Safe control action
        """
        # Get the nominal action from MPPI
        nominal_action = super().command(state)
        
        # If HJ reachability is not available, return the nominal action
        if self.hj_values is None or self.hj_solver is None:
            return nominal_action
        
        # Check if the chosen trajectory is safe
        state_np = state.cpu().numpy() if torch.is_tensor(state) else state
        
        # Get the chosen trajectory
        chosen_trajectory = self.get_chosen_trajectory(state_np)
        
        # Check each state in the trajectory for safety
        for t in range(len(chosen_trajectory)):
            traj_state = chosen_trajectory[t]
            is_safe, _, _ = self.hj_solver.check_if_safe(traj_state, self.hj_values)
            
            if not is_safe:
                # If any state in the trajectory is unsafe, compute a safe action
                action, _, _, _ = self.hj_solver.compute_safe_control(
                    state_np,
                    nominal_action.cpu().numpy(),
                    action_bounds=np.array([[-np.pi/4, np.pi/4], [-0.25, 0.25]]),
                    values=self.hj_values,
                    values_grad=self.hj_values_grad
                )
                return torch.tensor(action, device=self.d, dtype=self.dtype)
        
        # If the entire trajectory is safe, return the nominal action
        return nominal_action
    
    def get_chosen_trajectory(self, state):
        """
        Get the chosen trajectory based on the current control sequence.
        
        Args:
            state: Current state of the system
            
        Returns:
            list: List of states in the chosen trajectory
        """
        # Initialize trajectory with current state
        trajectory = [state]
        
        # Roll out the trajectory using the current control sequence
        for t in range(self.T):
            action = self.U[t].cpu().numpy()
            state = dubins_dynamics_tensor(
                torch.tensor(state[np.newaxis, :]), 
                torch.tensor(action[np.newaxis, :]), 
                self.dt
            )[0].cpu().numpy()
            trajectory.append(state)
            
        return trajectory


class DualGuardNavigator(Navigator):
    """
    Navigator that uses DualGuardMPPI for planning.
    """
    
    def __init__(self, hj_solver=None, safety_threshold=0.0, gradient_scale=1.0, **kwargs):
        # Store HJ solver and parameters before calling super().__init__
        self.hj_solver = hj_solver
        self.safety_threshold = safety_threshold
        self.gradient_scale = gradient_scale
        
        # Set planner_type to "dualguard_mppi"
        kwargs["planner_type"] = "dualguard_mppi"
        super().__init__(**kwargs)
        
    def set_hj_solver(self, hj_solver):
        """Set the HJ reachability solver."""
        self.hj_solver = hj_solver
        if self.planner_type == "dualguard_mppi" and self.planner is not None:
            self.planner.hj_solver = hj_solver
            
    def set_hj_values(self, values, values_grad=None):
        """Set the HJ reachability values and gradients."""
        if self.planner_type == "dualguard_mppi" and self.planner is not None:
            self.planner.set_hj_values(values, values_grad)
    
    def get_command(self):
        """Override to handle dualguard_mppi planner type."""
        x = self._state_torch[0]
        y = self._state_torch[1]
        dist_goal = torch.sqrt(
            (x - self._goal_torch[0]) ** 2 + (y - self._goal_torch[1]) ** 2
        )
        if dist_goal.item() < self._goal_thresh:
            return torch.tensor([0.0, 0.0], device=self.device, dtype=self.dtype)
        
        # Handle both mppi and dualguard_mppi planner types
        if self.planner_type in ["mppi", "dualguard_mppi"]:
            command = self.planner.command(self._state_torch)
            return command
        
        return None
            
    def _start_planner(self):
        """Override to create a DualGuardMPPI planner if planner_type is 'dualguard_mppi'."""
        if self.planner_type == "dualguard_mppi":
            mppi_config = self.make_mppi_config()
            mppi_config["hj_solver"] = self.hj_solver
            mppi_config["safety_threshold"] = self.safety_threshold
            mppi_config["gradient_scale"] = self.gradient_scale
            return DualGuardMPPI(**mppi_config)
        else:
            return super()._start_planner()