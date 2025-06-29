from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_field import LeggedRobotField


from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import numpy as np


class H1_41Robot(LeggedRobot):
    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        # rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # self.rigid_body_states_view = self.all_rigid_body_states.view(self.num_envs, -1, 13)
        # self.feet_state = self.rigid_body_states_view
        self.feet_state = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)

    # further modification
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    # def update_feet_state(self):
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 下面自动更新
        # self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        # self.feet_pos = self.feet_state[:, :, :3]
        # self.feet_vel = self.feet_state[:, :, 7:10]

    # 对post_physics_step_callback的进一步修改没问题
    def _post_physics_step_callback(self):
        # self.update_feet_state()

        period = 0.8
        offset = 0.5
        phase = (self.episode_length_buf * self.dt) % period / period
        self.leg_phase[:, 0] = phase
        self.leg_phase[:, 1] = (phase + offset) % 1
        # torch.cuda.empty_cache()

        return super()._post_physics_step_callback()

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
