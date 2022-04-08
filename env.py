from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch


"""
   Cartpole environment built on top of Isaac Gym.
   Based on the official implementation here: 
   https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/cartpole.py
"""


class Cartpole:
    def __init__(self, args):
        self.args = args

        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        # task-specific parameters
        self.num_obs = 4  # pole_angle + pole_vel + cart_vel + cart_pos
        self.num_act = 1  # force applied on the pole (-1 to 1)
        self.reset_dist = 3.0  # when to reset
        self.max_push_effort = 400.0  # the range of force applied to the cartpole
        self.max_episode_length = 500  # maximum episode length

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_dof = self.create_envs()
        self.dof_states = self.get_states_tensor()
        self.dof_pos = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_states.view(self.args.num_envs, self.num_dof, 2)[..., 1]

        # generate viewer for visualisation
        if not self.args.headless:
            self.viewer = self.create_viewer()

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        self.reset()

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = 2.5
        lower = gymapi.Vec3(0, 0, 0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # add cartpole asset
        asset_root = 'assets'
        asset_file = 'cartpole.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        # define cartpole pose
        pose = gymapi.Transform()
        pose.p.z = 1.0   # generate the cartpole 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # define cartpole dof properties
        dof_props = self.gym.get_asset_dof_properties(cartpole_asset)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0

        # generate environments
        envs = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add cartpole here in each environment
            cartpole_handle = self.gym.create_actor(env, cartpole_asset, pose, "cartpole", i, 1, 0)
            self.gym.set_actor_dof_properties(env, cartpole_handle, dof_props)

            envs.append(env)
        return envs, num_dof

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def get_states_tensor(self):
        # get dof state tensor (of cartpole)
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_states = dof_states.view(self.args.num_envs, self.num_obs)
        return dof_states

    def get_obs(self, env_ids=None):
        # get state observation from each environment id
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.obs_buf[env_ids] = self.dof_states[env_ids]

    def get_reward(self):
        self.reward_buf[:], self.reset_buf[:] = compute_cartpole_reward(self.dof_states,
                                                                        self.reset_dist,
                                                                        self.reset_buf,
                                                                        self.progress_buf,
                                                                        self.max_episode_length)

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.args.sim_device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.args.sim_device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset desired environments
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # clear up desired buffer states
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # refresh new observation after reset
        self.get_obs()

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # update viewer
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step(self, actions):
        # apply action
        actions_tensor = torch.zeros(self.args.num_envs * self.num_dof, device=self.args.sim_device)
        actions_tensor[::self.num_dof] = actions.squeeze(-1) * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        # simulate and render
        self.simulate()
        if not self.args.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()


# define reward function using JIT
@torch.jit.script
def compute_cartpole_reward(obs_buf, reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # retrieve each state from observation buffer
    cart_pos, cart_vel, pole_angle, pole_vel = torch.split(obs_buf, [1, 1, 1, 1], dim=1)

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reward[:, 0], reset[:, 0]
