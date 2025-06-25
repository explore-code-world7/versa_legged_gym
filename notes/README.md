
# initial step
1. 创建OnPolicyRunner()，
* camera用sim.no_camera确定

## LeggedRobot
* init_buffer: 用环境的状态信息，初始化所有buffer
* _prepare_reward_function：准备一系列奖励函数
* create_gournd_plane：设置地形
* create_envs：创建环境，设置asset，create_actor
* get_env_origins：utilization of torch.meshgrid
* select obs configurations
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
```