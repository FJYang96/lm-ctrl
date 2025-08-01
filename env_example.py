import numpy as np

from tqdm import tqdm
from gym_quadruped.quadruped_env import QuadrupedEnv
import time

robot_name = 'go2' # 'b2', 'go1', 'go2', 'hyqreal', 'mini_cheetah', 'aliengo'
terrain_type = 'flat' # 'flat', 'perlin'
sim_dt = 0.01 # seconds
sim_duration = 1 # seconds

################################################################################
# Create environment
################################################################################
print(f'Testing robot {robot_name} on terrain {terrain_type}')
env = QuadrupedEnv(
    robot=robot_name,
    scene=terrain_type,
    ref_base_lin_vel=(0.5, 1.0),  # pass a float for a fixed value
    ground_friction_coeff=(0.2, 1.5),  # pass a float for a fixed value
    base_vel_command_type='forward',  # "forward", "random", "forward+rotate", "human"
    # state_obs_names=...,  # Desired quantities in the 'state', see ALL_OBS in quadruped_env.py
    sim_dt=sim_dt,
)

################################################################################
# Print out the observation space
################################################################################
print("-"*20, "Observation space", "-"*20)
state = env.reset()
for obs_name, obs_val in state.items():
    print(f'{obs_name:20s}--\t{obs_val.shape}')

################################################################################
# Print out the action space
################################################################################
print("-"*20, "Action space", "-"*25)
print(env.action_space)

################################################################################
# Simulate and render
################################################################################
print("-"*20, "Simulating and rendering", "-"*13)
images = []
sim_times, render_times = [], []
for _ in tqdm(range(int(sim_duration / sim_dt))):
    import imageio

    # Step forward in the simulation
    start_time = time.time()
    action = env.action_space.sample() * 5  # Sample random action
    state, reward, is_terminated, is_truncated, info = env.step(action=action)
    sim_times.append(time.time() - start_time)

    # Render the environment into an image
    render_start_time = time.time()
    image = env.render(mode='rgb_array', tint_robot=True)
    render_times.append(time.time() - render_start_time)

    images.append(image)

fps = 1 / sim_dt
imageio.mimsave('test.mp4', images, fps=fps)
env.close()

print("-"*20, "Performance", "-"*20)
print(f'Sim time: {np.mean(sim_times):.3f} s, Render time: {np.mean(render_times):.3f} s')