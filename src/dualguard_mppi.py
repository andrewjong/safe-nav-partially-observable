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
    
    def _compute_rollout_costs(self, perturbed_actions):
        """
        Optimized version of rollout cost computation for better performance.
        
        Args:
            perturbed_actions: Actions to roll out
            
        Returns:
            cost_total: Total cost of each trajectory
            states: States visited in each trajectory
            actions: Actions taken in each trajectory
        """
        K, T = perturbed_actions.shape[0], perturbed_actions.shape[1] // self.nu
        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = torch.zeros(self.M, K, device=self.d, dtype=self.dtype)
        cost_var = torch.zeros(K, device=self.d, dtype=self.dtype)
        
        # Initialize state
        if hasattr(self, "init_cov"):
            eps = torch.randn(
                K, self.nx, device=self.d, dtype=self.dtype
            ) @ self.init_cov
            state = self.state.view(1, -1).repeat(K, 1) + eps
        else:
            state = self.state.view(1, -1).repeat(K, 1)
        
        # Repeat state for multiple rollouts
        state = state.repeat(self.M, 1, 1)
        
        # Pre-allocate tensors for states and actions
        states_list = []
        actions_list = []
        
        # Perform rollouts in parallel
        for t in range(T):
            # Scale and repeat actions
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            
            # Apply dynamics
            state = self._dynamics(state, u, t, self.dt)
            
            # Compute running cost
            c = self._running_cost(state, u, t)
            cost_samples += c
            
            # Compute variance if using multiple rollouts
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount**t)
            
            # Save states and actions
            states_list.append(state)
            actions_list.append(u)
        
        # Stack states and actions
        actions = torch.stack(actions_list, dim=-2)
        states = torch.stack(states_list, dim=-2)
        
        # Add terminal state cost if applicable
        if self.terminal_state_cost:
            c = self._terminal_state_cost(states[..., -1, :], actions[..., -1, :])
            cost_samples += c
        
        # Compute mean cost across rollouts
        cost_total += cost_samples.mean(dim=0)
        
        # Add variance cost
        cost_total += cost_var * self.rollout_var_cost
        
        return cost_total, states, actions
    
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
            
            # Create time scaling factors for all time steps at once
            time_scales = 1.0 / (1.0 + 0.1 * torch.arange(self.T, device=self.d, dtype=self.dtype))
            
            # Compute safety bias for all time steps at once
            safety_bias = self.gradient_scale * lhs_tensor.unsqueeze(0).unsqueeze(0) * time_scales.unsqueeze(1)
            
            # Apply the guidance to all noise samples at once
            self.noise += safety_bias
    
    def _apply_safety_cost_to_rollouts(self):
        """
        Apply an additional safety cost to the rollouts based on HJ values.
        This penalizes trajectories that enter unsafe regions.
        Optimized with batch processing for better performance.
        """
        # Get the states from the rollouts
        # states shape: M x K x T x nx
        M, K, T, nx = self.states.shape
        
        # Initialize safety cost
        safety_cost = torch.zeros((M, K), device=self.d, dtype=self.dtype)
        
        # Create time penalty factors for all time steps
        time_penalties = 10.0 / (1.0 + 0.1 * np.arange(T))
        
        # Process in batches to avoid memory issues
        batch_size = 100  # Adjust based on available memory
        
        for m in range(M):
            for k_batch in range(0, K, batch_size):
                k_end = min(k_batch + batch_size, K)
                k_range = range(k_batch, k_end)
                
                for t in range(T):
                    # Get states for this batch and time step
                    batch_states = self.states[m, k_batch:k_end, t].cpu().numpy()
                    
                    # Check safety for all states in the batch
                    for i, k in enumerate(k_range):
                        state = batch_states[i]
                        is_safe, _, _ = self.hj_solver.check_if_safe(state, self.hj_values)
                        
                        if not is_safe:
                            safety_cost[m, k] += time_penalties[t] * 100.0
        
        # Add the safety cost to the total cost
        self.cost_total += safety_cost.mean(dim=0)
    
    def command(self, state):
        """
        Override the command method to ensure the final trajectory is safe.
        Optimized for better performance.
        
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
        
        # Convert trajectory to numpy array for faster processing
        trajectory_array = np.array(chosen_trajectory)
        
        # Check if any state in the trajectory is unsafe
        is_unsafe = False
        
        # Process trajectory in batches for efficiency
        batch_size = 10  # Adjust based on performance
        num_states = len(trajectory_array)
        
        for i in range(0, num_states, batch_size):
            end_idx = min(i + batch_size, num_states)
            batch = trajectory_array[i:end_idx]
            
            # Check each state in the batch
            for j in range(len(batch)):
                traj_state = batch[j]
                is_safe, _, _ = self.hj_solver.check_if_safe(traj_state, self.hj_values)
                
                if not is_safe:
                    is_unsafe = True
                    break
            
            if is_unsafe:
                break
        
        if is_unsafe:
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
        Optimized for better performance.
        
        Args:
            state: Current state of the system
            
        Returns:
            list: List of states in the chosen trajectory
        """
        # Initialize trajectory with current state
        trajectory = [state]
        current_state = state.copy()
        
        # Convert control sequence to numpy for faster processing
        U_np = self.U.cpu().numpy()
        
        # Pre-allocate tensor for dynamics computation
        state_tensor = torch.tensor(current_state[np.newaxis, :], dtype=self.dtype, device=self.d)
        
        # Roll out the trajectory using the current control sequence
        for t in range(self.T):
            action = U_np[t]
            action_tensor = torch.tensor(action[np.newaxis, :], dtype=self.dtype, device=self.d)
            
            # Update state using dynamics
            state_tensor = dubins_dynamics_tensor(state_tensor, action_tensor, self.dt)
            current_state = state_tensor[0].cpu().numpy()
            trajectory.append(current_state.copy())
            
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
        
    def get_sampled_trajectories(self):
        """Override to handle dualguard_mppi planner type."""
        if self.planner_type in ["mppi", "dualguard_mppi"]:
            # states: torch.tensor, shape(M, K, T, nx)
            trajectories = self.planner.states
            M, K, T, nx = trajectories.shape
            return trajectories.view(M * K, T, nx)
        return None
            
    def get_chosen_trajectory(self):
        """
        Override to handle dualguard_mppi planner type.
        
        Returns:
            torch.Tensor: Tensor of shape (T+1, nx) containing the chosen trajectory
        """
        if self.planner_type in ["mppi", "dualguard_mppi"]:
            # Start with current state
            state = self._state_torch.clone().unsqueeze(0)  # Shape: (1, nx)
            trajectory = [state.squeeze(0)]
            
            # Roll out the trajectory using the current control sequence
            for t in range(self.planner.T):
                action = self.planner.U[t].unsqueeze(0)  # Shape: (1, nu)
                state = dubins_dynamics_tensor(state, action, self.dt)
                trajectory.append(state.squeeze(0))
                
            return torch.stack(trajectory)
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