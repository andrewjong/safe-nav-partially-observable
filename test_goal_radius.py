import posggym
import numpy as np

# Create environment with default goal_radius (same as agent_radius)
env1 = posggym.make("DrivingContinuous-v0", world="6x6", num_agents=2)
print(f"Default goal_radius: {env1.model.world.goal_radius}")
print(f"Default agent_radius: {env1.model.world.agent_radius}")

# Create environment with custom goal_radius
env2 = posggym.make("DrivingContinuous-v0", world="6x6", num_agents=2, goal_radius=0.3)
print(f"Custom goal_radius: {env2.model.world.goal_radius}")
print(f"Custom agent_radius: {env2.model.world.agent_radius}")

# Test that goal checking uses goal_radius instead of agent_radius
obs, info = env2.reset()
print("Initial state:")
for agent_id, agent_state in enumerate(env2.state):
    dest_coord = agent_state.dest_coord
    agent_pos = agent_state.body[:2]
    dist = np.linalg.norm(np.array(agent_pos) - np.array(dest_coord))
    print(f"Agent {agent_id} - Distance to goal: {dist:.4f}, Goal reached: {bool(agent_state.status[0])}")

# Manually set agent position close to goal but outside agent_radius
# This should show the agent as having reached the goal if goal_radius is working
print("\nManually setting agent position close to goal:")
for agent_id, agent_state in enumerate(env2.state):
    dest_coord = agent_state.dest_coord
    # Set position to be at a distance between agent_radius and goal_radius
    dist = 0.2  # Between agent_radius (0.1) and goal_radius (0.3)
    direction = np.random.rand(2)
    direction = direction / np.linalg.norm(direction)
    new_pos = np.array(dest_coord) - dist * direction
    
    # Update agent position
    env2.state[agent_id] = env2.state[agent_id]._replace(
        body=np.array([new_pos[0], new_pos[1], 0, 0, 0], dtype=np.float32),
        status=np.array([0, 0], dtype=np.int8)  # Reset status
    )

# Step the environment to update the status
obs, rewards, terminated, truncated, info = env2.step({0: np.array([0, 0]), 1: np.array([0, 0])})

# Check if agents reached their goals
print("After moving agents:")
for agent_id, agent_state in enumerate(env2.state):
    dest_coord = agent_state.dest_coord
    agent_pos = agent_state.body[:2]
    dist = np.linalg.norm(np.array(agent_pos) - np.array(dest_coord))
    print(f"Agent {agent_id} - Distance to goal: {dist:.4f}, Goal reached: {bool(agent_state.status[0])}")
    print(f"  - Is distance <= agent_radius? {dist <= env2.model.world.agent_radius}")
    print(f"  - Is distance <= goal_radius? {dist <= env2.model.world.goal_radius}")