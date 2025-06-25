from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_field import LeggedRobotField

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import numpy as np


class H1_41Robot(LeggedRobotField):

    # def _create_envs(self):
    #     """ Creates environments:
    #          1. loads the robot URDF/MJCF asset,
    #          2. For each environment
    #             2.1 creates the environment,
    #             2.2 calls DOF and Rigid shape properties callbacks,
    #             2.3 create actor with these properties and add them to the env
    #          3. Store indices of different bodies of the robot
    #     """
    #     asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    #     asset_root = os.path.dirname(asset_path)
    #     asset_file = os.path.basename(asset_path)
    #
    #     asset_options = gymapi.AssetOptions()
    #     asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
    #     asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
    #     asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
    #     asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
    #     asset_options.fix_base_link = self.cfg.asset.fix_base_link
    #     asset_options.density = self.cfg.asset.density
    #     asset_options.angular_damping = self.cfg.asset.angular_damping
    #     asset_options.linear_damping = self.cfg.asset.linear_damping
    #     asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
    #     asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
    #     asset_options.armature = self.cfg.asset.armature
    #     asset_options.thickness = self.cfg.asset.thickness
    #     asset_options.disable_gravity = self.cfg.asset.disable_gravity
    #
    #     robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
    #     self.num_dof = self.gym.get_asset_dof_count(robot_asset)
    #     self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
    #     dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
    #     rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
    #
    #     # save body names from the asset
    #     body_names = self.gym.get_asset_rigid_body_names(robot_asset)
    #     self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    #     self.num_bodies = len(body_names)
    #     self.num_dofs = len(self.dof_names)
    #     feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
    #     penalized_contact_names = []
    #     for name in self.cfg.asset.penalize_contacts_on:
    #         penalized_contact_names.extend([s for s in body_names if name in s])
    #     termination_contact_names = []
    #     for name in self.cfg.asset.terminate_after_contacts_on:
    #         termination_contact_names.extend([s for s in body_names if name in s])
    #
    #
    #     # print(f"self.dof_names = {self.dof_names[:23]}")
    #     # print(f"self.dof_names = {self.dof_names[23:]}")
    #     # limit self.num_dofs
    #     '''
    #     # Q: 关节并不是按dof排列的
    #     # 选取手指的指定关节即可;
    #     # 按手指，找到手指对应的indexes;
    #     # 将手指indexes对应的dof设为
    #     '''
    #     finger_names = []
    #     for finger in self.cfg.asset.finger_name:
    #         _finger_names = [s for s in self.dof_names if finger in s]
    #         finger_names.extend(_finger_names)
    #
    #     self.finger_indices = [self.gym.find_asset_dof_index(robot_asset, name) for name in finger_names]
    #     self.non_finger_indices = [j for j in range(self.num_dofs) if j not in self.finger_indices]
    #
    #     # for i in self.finger_indices:
    #     #     name = self.dof_names[i]
    #     #     angle = self.cfg.init_state.default_joint_angles[name]
    #     #     # dof_props_asset[i][1] = angle
    #     #     # dof_props_asset[i][2] = angle
    #     #     # dof_props_asset[i][6] = 0.0
    #     #     # dof_props_asset[i][7] = 0.0
    #     #     # velocity+effor can't be zero
    #     #     dof_props_asset["lower"][i] = angle
    #     #     dof_props_asset["upper"][i] = angle
    #     #     dof_props_asset["velocity"][i] = 0.01
    #     #     dof_props_asset["effort"][i] = 0.01
    #
    #     base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
    #     self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
    #     start_pose = gymapi.Transform()
    #     start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
    #
    #     self._get_env_origins()
    #     env_lower = gymapi.Vec3(0., 0., 0.)
    #     env_upper = gymapi.Vec3(0., 0., 0.)
    #     self.actor_handles = []
    #     self.envs = []
    #     for i in range(self.num_envs):
    #         # create env instance
    #         env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
    #         pos = self.env_origins[i].clone()
    #         pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
    #         start_pose.p = gymapi.Vec3(*pos)
    #
    #         rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
    #         self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
    #         actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
    #                                              self.cfg.asset.self_collisions, 0)
    #         dof_props = self._process_dof_props(dof_props_asset, i)
    #         # print(f"H1_41Robot, line 90, dof_props={dof_props}")
    #         self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
    #         body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
    #         body_props = self._process_rigid_body_props(body_props, i)
    #         self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
    #         self.envs.append(env_handle)
    #         self.actor_handles.append(actor_handle)
    #
    #     self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
    #     for i in range(len(feet_names)):
    #         self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
    #                                                                      feet_names[i])
    #
    #     self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
    #                                                  requires_grad=False)
    #     for i in range(len(penalized_contact_names)):
    #         self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
    #                                                                                   self.actor_handles[0],
    #                                                                                   penalized_contact_names[i])
    #
    #     self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
    #                                                    device=self.device, requires_grad=False)
    #     for i in range(len(termination_contact_names)):
    #         self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
    #                                                                                     self.actor_handles[0],
    #                                                                                     termination_contact_names[i])

    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure
    #
    #     Args:
    #         cfg (Dict): Environment config file
    #
    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     noise_vec = torch.zeros_like(self.obs_buf[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[3:6] = noise_scales.gravity * noise_level
    #     noise_vec[6:9] = 0. # commands
    #     noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
    #     noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
    #
    #     return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    # further modification
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    # 对post_physics_step_callback的进一步修改没问题
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        return super()._post_physics_step_callback()

    # def compute_observations(self):
    #     """ Computes observations
    #     """
    #     sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
    #     cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
    #     # print(self.dof_pos.shape)
    #     # print(self.dof_pos[0,23:])
    #     self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 self.commands[:, :3] * self.commands_scale,
    #                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 self.actions,
    #                                 # sin_phase, comand f
    #                                 # cos_phase
    #                                 ),dim=-1)
    #     self.privileged_obs_buf = torch.cat((
    #                                 self.base_lin_vel * self.obs_scales.lin_vel,
    #                                 self.base_ang_vel  * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 self.commands[:, :3] * self.commands_scale,
    #                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 self.actions,
    #                                 # sin_phase,
    #                                 # cos_phase
    #                                 ),dim=-1)
    #
    #     # add perceptive inputs if not blind
    #     # add noise if needed
    #     if self.add_noise:
    #         self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    ##### reward designs #####

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, [0, 2, 6, 8]]), dim=1)
