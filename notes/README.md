# initial step

1. 创建OnPolicyRunner()，

## LeggedRobot

* init_buffer: 用环境的状态信息，初始化所有buffer

* _prepare_reward_function：准备一系列奖励函数

* create_gournd_plane：设置地形

* create_envs：创建环境，设置asset，create_actor

* get_env_origins：utilization of torch.meshgrid

```python
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols)) # xx:[num_row, num_col]
        spacing = self.cfg.env.env_spacing
        print(f"env_spacing = {spacing}")
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.
```

> cfg就是一个字典，很好调用的

* get_noise_scale_vec：创建一个noise_vec，用来扰乱observation

* reset_root_states：用root_states更新环境属性

```python
self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor
    (self.root_states),gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
```

* 类似reset_dof_states, 

* resample_commands：在范围内随机选取命令

### process props

* _process_rigid_body_props：randomize base mass

* _process_dof_props：

* _process_rigid_shape_props：

***

* 如何获取observation，如何设置force的？

### compute torques

```python
torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
```
## force

```python
net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
```

## sensors

```python
        self.sensor_handles = []
        sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
        self.sensor_handles.append(sensor_handle_dict)

    def _create_sensors(self, env_handle= None, actor_handle= None): 
        if "forward_depth" in all_obs_components or "forward_color" in all_obs_components:
            camera_handle = self._create_onboard_camera(env_handle, actor_handle, "forward_camera")
            sensor_handle_dict["forward_camera"] = camera_handle

    def _create_onboard_camera(self, env_handle, actor_handle, sensor_name):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = getattr(self.cfg.sensor, sensor_name).resolution[0]
        camera_props.width = getattr(self.cfg.sensor, sensor_name).resolution[1]
        ...
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        ...
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
```
* get image tensor(binded)
```python
    def _init_sensor_buffers(self, env_i, env_handle):
        if "forward_depth" in self.all_obs_components:
            self.sensor_tensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
                self.gym.get_camera_image_gpu_tensor(
                    self.sim,
                    env_handle,
                    self.sensor_handles[env_i]["forward_camera"],
                    gymapi.IMAGE_DEPTH,
            )))
```
* digest gpu tensor in simulation(in post_physics_step/)
```python
    def compute_observations(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
```
* concatenate into obs
```python
    def _get_forward_depth_obs(self, privileged= False):
        return torch.stack(self.sensor_tensor_dict["forward_depth"]).flatten(start_dim= 1)
```
