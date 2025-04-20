import numpy as np

import posggym
from reachability.warm_start_solver import (WarmStartSolver,
                                            WarmStartSolverConfig)
from src.mppi import Navigator, dubins_dynamics_tensor


def main():
    env = posggym.make('DrivingContinuous-v0', world="14x14RoundAbout", num_agents=1, render_mode="human")

    solver = WarmStartSolver(
        config=WarmStartSolverConfig(
            system_name="dubins3d",
            domain_cells=[30, 40, 40],
            domain=[[0, 0, 0], [14, 14, 2*np.pi]],
            mode="brt",
            accuracy="medium",
            converged_values=None,
            until_convergent=False,
            print_progress=False,
        )
    )

    # nom_controller = Navigator()
    # nom_controller.set_odom(state[:2],state[2])
    # nom_controller.set_map(builder.failure_map, [30, 40], [0, 0], 1.0)
    # nom_controller.set_goal(list(env_params.goal_location))

    observations, infos = env.reset()

    for _ in range(300):
        actions = {i: env.action_spaces[i].sample() for i in env.agents}
        observations, rewards, terminations, truncations, all_done, infos = env.step(actions)
        env.render()

        if all_done:
            observations, infos = env.reset()

    env.close()


if __name__ == "__main__":
    main()