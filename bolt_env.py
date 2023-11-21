from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import torch

class BoltEnv:
    def __init__(self, args):
        self.args = args

        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.dt = 1 / 60.
        sim_params.use_gpu_pipeline = True
        sim_params.physx.use_gpu = True

        self.num_dof_obs = 12
        self.num_rigid_obs = 13
        self.num_obs = self.num_dof_obs + self.num_rigid_obs + 2 #2 for contactss
        self.num_act = 6
        self.max_episode_length = 1000
        self.max_effort = 10
        self.reset_height = 0.1

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.forces_buf = torch.zeros((self.args.num_envs, self.num_act), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_dof = self.create_envs()

        #get state tensors
        self.dof_states = self.get_dof_states_tensor()
        self.dof_pos = self.dof_states[..., 0]
        self.dof_vel = self.dof_states[..., 1]

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.args.sim_device, requires_grad=False)

        self.rigid_states = self.get_rigid_states_tensor()
        self.base_lin_pos = self.rigid_states[..., 0, 0:3]
        self.base_lin_vel = self.rigid_states[..., 0, 7:10]
        self.base_ang_pos = self.rigid_states[..., 0, 3:7]
        self.base_ang_vel = self.rigid_states[..., 0, 10:13]

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)

        self.contact_forces = self.get_contact_force_tensor()

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

        # add bolt asset
        asset_root = 'assets'
        asset_file = 'bolt.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        bolt_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_dof = self.gym.get_asset_dof_count(bolt_asset)

        # define bolt pose
        pose = gymapi.Transform()
        pose.p.z = 0.5  
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)        

        self.base_init_state= torch.tensor([0.0, 0.0, pose.p.z] + [0.0, 0.0, 0.0, 1.0] + 6 * [0], device=self.args.sim_device)

        # define bolt dof properties
        dof_props = self.gym.get_asset_dof_properties(bolt_asset)
        dof_props['driveMode'] = gymapi.DOF_MODE_EFFORT

        envs = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add bolt here in each environment
            bolt_handle = self.gym.create_actor(env, bolt_asset, pose, "bolt", i, 1, 0)
            self.gym.set_actor_dof_properties(env, bolt_handle, dof_props)

            envs.append(env)
        return envs, num_dof
    
    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer
    
    def get_dof_states_tensor(self):
        # get dof state tensor (of bolt)
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_states = dof_states.view(self.args.num_envs, self.num_dof, 2)
        return dof_states
    
    def get_rigid_states_tensor(self):
        # get rigid state tensor (of bolt)
        _rigid_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_states = gymtorch.wrap_tensor(_rigid_states)
        rigid_states = rigid_states.view(self.args.num_envs, 9, 13) #9 rigid bodies, 13 states
        return rigid_states
    
    def get_contact_force_tensor(self):
        # get force sensor tensor (of bolt)
        _contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_forces = gymtorch.wrap_tensor(_contact_forces)
        contact_forces = contact_forces.view(self.args.num_envs, 9, 3)
        return contact_forces


    def get_obs(self, env_ids=None):
    # get state observation from each environment id
        if env_ids is None:
            env_ids = torch.arange(self.args.num_envs, device=self.args.sim_device)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        contacts = self.contact_forces[..., 7:9, 2] > 0.1
        
        self.obs_buf[env_ids] = torch.cat((self.base_lin_pos[env_ids], self.base_lin_vel[env_ids], \
                                           self.base_ang_pos[env_ids], self.base_ang_vel[env_ids], \
                                           self.dof_pos[env_ids], self.dof_vel[env_ids], contacts[env_ids]), dim=1)

    def get_reward(self):
        self.reward_buf[:], self.reset_buf[:] = compute_bolt_reward(self.obs_buf,
                                                                        self.forces_buf,
                                                                        self.reset_height,
                                                                        self.reset_buf,
                                                                        self.progress_buf,
                                                                        self.max_episode_length)
  
    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities for dof
        dof_positions = self.default_dof_pos # + 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.args.sim_device) - 0.5)
        dof_velocities = torch.zeros((len(env_ids), self.num_dof), device=self.args.sim_device)

        self.dof_pos[env_ids] = dof_positions 
        self.dof_vel[env_ids] = dof_velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset DOFs
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_states),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        # reset rigid bodies
        self.root_states[env_ids] = self.base_init_state
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
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
        # apply actions
        forces = actions * self.max_effort
        self.forces_buf = forces
        forces = gymtorch.unwrap_tensor(forces)
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
def compute_bolt_reward(obs_buf, forces_buf, reset_height, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # retrieve each state from observation buffer
    base_lin_pos, base_lin_vel, base_angle_pos, base_angle_vel, joints_angle, joints_vel, contacts = torch.split(obs_buf, [3, 3, 4, 3, 6, 6, 2], dim=1)

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 10 * torch.ones_like(base_lin_pos[:, 2]) #1 - 0.1*torch.norm(base_lin_vel) - 0.1*torch.norm(base_angle_vel)#+ -5 * torch.abs(0.4 - base_lin_pos[:, 2]) #+ 0.2*torch.sum(contacts)
    # adjust reward for reset agents
    reward = torch.where((base_lin_pos[:, 2] < reset_height), torch.ones_like(reward) * -1000, reward)  
    reward = torch.where((base_lin_pos[:, 2] < 0.35), torch.ones_like(reward) * -10, reward)   
    reward = torch.where((base_lin_pos[:, 2] < 0.3), torch.ones_like(reward) * -100, reward)
    reward = torch.where((base_lin_pos[:, 2] < 0.2), torch.ones_like(reward) * -500, reward)
    reset = torch.where((base_lin_pos[:, 2] < reset_height), torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    return reward, reset