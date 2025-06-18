
# leggedrobot
1. _get_xxx: acquire intermittent values
2. get_obs_segment_from_components: 从观察中组合出若干部分，用作
    * get_num_obs_from_components: compute num of obs from components
    * privileged_obs_segments
    * obs_sgement
    * _get_noise_scale_vec: Sets a vector used to scale the noise added to the observations.
        具体实现调用write_xxx_noise
    * _draw_height_measurements_vis: Draws height measurements as animated depth image
* _get_obs_from_components: 用来生成obs_buf+ has_attri_buf
3. _process_dof_props: modify dof_props of .urdf manually by config.py
4. _process_rigid_body_props: 随机化质心
5. post_physics_step_callback: add height sensor, push robots, 
6. _resample_commands: cut down small commands



9. get_noise_scale_vec: 针对obs_noise进行更新
* noise_vec's role is to update obs_buf
10. _init_buffers:
## sample volume
11. _init_body_volume_points: 
12. _init_volume_sample_points: 
13. refresh_volume_sample_points: 
* 对body_volume_points进行变换, outputs points_positions_w, points_velocity_w
```python
def _get_target_pos_vel(self, target_link_indices, target_local_pos, domain= gymapi.DOMAIN_SIM):
     if domain == gymapi.DOMAIN_SIM:
         target_body_states = self.all_rigid_body_states[target_link_indices].view(self.num_envs, -1, 13)
     elif domain == gymapi.DOMAIN_ENV:
         # NOTE: maybe also acceping DOMAIN_ACTOR, but do this at your own risk
         target_body_states = self.all_rigid_body_states.view(self.num_envs, -1, 13)[:, target_link_indices]
     else:
         raise ValueError(f"Unsupported domain: {domain}")
     # shape: (n_envs, n_targets_per_robot, 3)
     target_pos = target_body_states[:, :, 0:3]
     # shape: (n_envs, n_targets_per_robot, 4)
     target_quat = target_body_states[:, :, 3:7]
     # shape: (n_envs * n_targets_per_robot, 3)
     target_pos_world_ = tf_apply(
         target_quat.view(-1, 4),
         target_pos.view(-1, 3),
         target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape).reshape(-1, 3), # using reshape because of contiguous issue
     )
     # shape: (n_envs, n_targets_per_robot, 3)
     target_pos_world = target_pos_world_.view(self.num_envs, -1, 3)
     # shape: (n_envs, n_targets_per_robot, 3)
     # NOTE: assuming the angular velocity here is the same as the time derivative of the axis-angle
     target_vel_world = torch.cross(
         target_body_states[:, :, 10:13],
         target_local_pos.unsqueeze(0).expand(self.num_envs, *target_local_pos.shape),
         dim= -1,
     )
     return target_pos_world, target_vel_world

* 这个是怎么采出来的?

* 这个是怎么使用的?
    1. legged_gym/legged_gym/envs/base/legged_robot_field.py
    2. legged_gym/legged_gym/utils/terrain/barrier_track.py
```


14. _init_sensor_buffers：collect data from camera sensors;
15. reset_buffers: refreshing buffers;
16. _create_sensors: 
17. _create_onboard_camera: 
18. _create_npc: 
19. _create_terrain: 
20. a series of vis: 
21. a series of get_xxx(height)
22. _fill_extras:


* copy homework to cope with trifle lists
* use param configs to tackle with xxx
