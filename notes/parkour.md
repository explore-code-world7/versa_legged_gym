* env的函数实现，对应配置文件

# Leggedrobot
## get partial observations
1. _get_xxx: acquire intermittent values
2. get_obs_segment_from_components: 从观察中组合出若干部分，用作
    * get_num_obs_from_components: compute num of obs from components
    * privileged_obs_segments
    * obs_sgement
    * _get_noise_scale_vec: Sets a vector used to scale the noise added to the observations.
        具体实现调用write_xxx_noise
    * _draw_height_measurements_vis: Draws height measurements as animated depth image
* cfg筛选观察
```python

```
* _get_obs_from_components: 用来生成obs_buf+ has_attri_buf

## Process step
3. _process_dof_props: modify dof_props of .urdf manually by config.py

4. _process_rigid_body_props: 随机化质心
```python
domain_rand.randomize_base_mass
```
5. post_physics_step_callback: add height sensor, push robots, 
6. _resample_commands: cut down small commands
7. _update_terrain_curriculum: in reset_idx


## add noise to observations
9. get_noise_scale_vec: 针对obs_noise进行更新
```python
noise
```
* _write_xxx: write vector of noises
10. _init_buffers:


## add volume points
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


## append sensors& terrains
16. _create_sensors: 
```python
    class sensor:
        class forward_camera:
            resolution = [16, 16]
            position = [0.26, 0., 0.03] # position in base_link
            rotation = [0., 0., 0.] # ZYX Euler angle in base_link
    
        class proprioception:
            delay_action_obs = False
            latency_range = [0.0, 0.0]
            latency_resampling_time = 2.0 # [s]
```
* fix camera to location
```python
transform = gymapi.Transform()
transform.p = (x,y,z)
transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
gym.set_camera_transform(camera_handle, env, transform)
gym.set_camera_location(camera_handle, env, gymapi.Vec3(x,y,z), gymapi.Vec3(tx,ty,yz))
```
* fix camera to body
```
local_transform = gymapi.Transform()
local_transform.p = (x,y,z)
local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
gym.attach_camera_to_body(camera_handle, env, body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
```
* save rgb
```python
for i in range(num_envs):
  # write tensor to image
  fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
  cam_img = rgbd_tensors[i].cpu().numpy()
  cam_img[cam_img == 165] = 255
  imageio.imwrite(fname, cam_img)
```
* write depth
```python
  # 不括号，会判断frame_no == (0|restart[i])
  if (frame_no == 1) | restart[i]:
      if perform_test:
          total_grasp_time -=1
      # print(f"total_grasp_time ={total_grasp_time}")
      com_img = torch.zeros(224, 224, 4).to(device)
      # cam_img = cam_tensors[i]
      # depth_img = depth_tensors[i]
      rgb_img = rgbd_tensors[i]        #rgbd
      depth_img = depth_tensors[i]
      depth_img2 = depth_img.cpu().numpy()       #这创建了一个新的对象,不必先clone()
      # -inf implies no depth value, set it to zero. output will be black.
      depth_img2[depth_img2 == -np.inf] = 0

      # clamp depth image to 10 meters to make output image human friendly
      depth_img2[depth_img2 < -10] = -10
      norm_depth_img = 255.0*(depth_img2/np.min(depth_img2 - 1e-4))
      normalized_depth_image = Image.fromarray(norm_depth_img.astype(np.uint8), mode="L")
      normalized_depth_image.save(os.path.join(predict_dir,f'frameno_{frame_no}_env_{i}.jpg'))
      np.savetxt(os.path.join(predict_dir,f'frameno_{frame_no}_env_{i}.txt'), depth_img.cpu().numpy(), fmt='%f')  # fmt='%d' 表示以整数格式写入
```

17. _create_onboard_camera:
18. _create_npc: 
19. _create_terrain:

20. a series of vis: 
* draw volume points function on measured heights and robot rigid body volume points
```python
    def _draw_volume_sample_points_vis(self):
        self.refresh_volume_sample_points()
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 4, 4, None, color=(0., 1., 0.))
        for env_idx in range(self.num_envs):
            for point_idx in range(self.volume_sample_points.shape[1]):
                sphere_pose = gymapi.Transform(gymapi.Vec3(*self.volume_sample_points[env_idx, point_idx]), r= None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose)
```
21. a series of get_xxx(height)
22. _fill_extras: rewrite extra appendings

## extra rewards

# Extensions
## legged_robot_field
1. fill_extras updates terrain info;
2. init_buffers, reset_buffers updates gravity infos;
* 为什么要进行这些更新，怎么更新?
* 想加某个状态量/条件量，在init/reset中添加；
* 
* 添加了噪声，推理下， 在get_observation中应用;
* 添加重力观察，reset_buffers， get_observations要修改;
3. check_termination: 检查各种中间量

4. get_terrain_curriculum_move:
* update difficulty of terrains;

5. _get_xxx_obs:
* block: get_engaging_block_distance and type;
* sidewall: get distance

6. reward_xxx:

## legged_robot_noisy
* 代码一旦加了很多功能，类的数据结构和逻辑会混乱
1. clip_position_action_by_torque_limit: 
