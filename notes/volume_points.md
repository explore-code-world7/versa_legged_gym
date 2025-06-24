* 实时maintain volume points来计算penetration dept
* 计算到地形表面的最大距离，表面可用周围高度场表示

# legged_robot/field
* refresh_volume_sample_points
```python
    def refresh_volume_sample_points(self):
        if self.volume_sample_points_refreshed:
            # use `volume_sample_points_refreshed` to avoid repeated computation
            return
        sample_points_start_idx = 0
        # search out measured volume points at assigned positions
        for body_idx, body_measure_name in enumerate(self.body_measure_name_order):
            volume_points = self.body_volume_points[body_measure_name] # (n_points, 3)
            num_volume_points = volume_points.shape[0]
            rigid_body_index = self.body_sample_indices[body_idx:body_idx+1] # size: torch.Size([1])
            point_positions_w, point_velocities_w = self._get_target_pos_vel(
                rigid_body_index.expand(num_volume_points,),
                volume_points,
                domain= gymapi.DOMAIN_ENV,
            )
            self.volume_sample_points_vel[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_velocities_w
            self.volume_sample_points[
                :,
                sample_points_start_idx: sample_points_start_idx + num_volume_points,
            ] = point_positions_w
            sample_points_start_idx += num_volume_points
        self.volume_sample_points_refreshed = True
```

* 作用：维护volume_sample_points_vel和volume__sample_points
## data stureture illustration
* self.body_volume_points: coodinate of mearsure points under robot coordination
* self.body_sample_indices: indices in sim.actor.rigid_body of all measure ponits in cfg.py
* self.volume_sample_points. self.volume_sample_points_vel denotes position and velocity of all mearured points
* get_target_pos_vel: 用self.all_rigid_body_states获取机器人当前的transform&rotation, 得到当前测量点在global coordination下的坐标;
以及在global coordination下的角速度(用叉积计算)
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
```


# terrain
* 阅读复杂代码，首要整理类的成员变量的含义，和初始化函数
## track_kwargs参数解析
* options: obstacle options
* self. track_block_resolution: 在每个块的分辨率x,y轴
* self.n_blocks_per_track: number of obstacles per track
* engaging_next_threshold: if > 0, engaging_next is based on this threshold instead of track_block_length/2. Make sure the obstacle is not too long.
* engagint_finish_threshold: an obstacle is considered finished only if the last volume point is this amount away from the block origin.
* well_height, wall_thickness: 墙的高度和厚度;

## ds
* self.track_resolution: 分辨率 of track
* self.track_info_map: [num_rows, num_cols, n_block_per_track, 1+self.block_info_dim],
  * param category: track_id, obstacle_depth, obstacle_critical_params
* self.track_width_map: [num_rows,num_cols]
* self.block_starting_height_map: [num_rows, num_cols, n_blocks_per_stack]
> each track denotes serveral block, there is a obstacle at each block, a track consist of a connected track_blocks
> so a track in infact denotes an env
* self.env_block_length, self.env_length, self.env_width:
* self.engating_next_min_forward_distance: 进入到下一个obstacle的最小前进距离；不同block，同一track是倍墙分割出来的
* 每个track_block的示意图
```python
        """ All track blocks are defined as follows
            +-----------------------+
            |xxxxxxxxxxxxxxxxxxxxxxx|track wall
            |xxxxxxxxxxxxxxxxxxxxxxx|
            |xxxxxxxxxxxxxxxxxxxxxxx|
            |                       |
            |                       |
            |                       |
            |                       |
            * (env origin)          |
            |                       |
            |                       | ^+y
            |                       | |
            |xxxxxxxxxxxxxxxxxxxxxxx| |
            |xxxxxxxxxxxxxxxxxxxxxxx| |
            |xxxxxxxxxxxxxxxxxxxxxxx| |         +x
            +-----------------------+ +---------->
```
* get_staring_track: 构造上面的墙，中间为海平面平地
```python
track_heighfield_template[:, :np.ceil(
    wall_thickness / self.cfg.horizontal_scale
).astype(int)] += ( \
    np.random.uniform(*self.track_kwargs["wall_height"]) \
    if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
    else self.track_kwargs["wall_height"] \
) / self.cfg.vertical_scale
track_heighfield_template[:, -np.ceil(
    wall_thickness / self.cfg.horizontal_scale
).astype(int):] += ( \
    np.random.uniform(*self.track_kwargs["wall_height"]) \
    if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
    else self.track_kwargs["wall_height"] \
) / self.cfg.vertical_scale
```
* get_jump_track: 没有设置墙，墙的高度默认为0,中间为障碍高度
```python
if not heightfield_noise is None:
    track_heightfield = heightfield_template + heightfield_noise
else:
    track_heightfield = heightfield_template.copy()
if not virtual and height_value > 0.:
    track_heightfield[
        1:,
        wall_thickness_px: -wall_thickness_px,
    ] += height_value
if height_value < 0.:
    track_heightfield[
        (0 if virtual else depth_px):,
        max(0, wall_thickness_px-1): min(-1, -wall_thickness_px+1),
    ] += height_value
```
* depth_px对应obstacle的x轴宽度；
* wall_thickness对应y轴宽度
* 真有问题，手动设置height_field的x轴坐标即可;

## func illustration
* 什么是starting_trimesh?
1. 之后200行-1053行均为构造地形的track_info 
2. build_height_raw: build heightfield and heightsamples with border
3. add_track_to_sim:
* 把trimesh和heightfield_raw添加到sim，
* 同时维护self.track_info_map, self.blocking_starting_height_map
```python
for obstacle_idx, obstacle_selection in enumerate(obstacle_order):
    obstacle_name = self.track_kwargs["options"][obstacle_selection]
    obstacle_id = self.track_options_id_dict[obstacle_name]
    # call method to generate trimesh and heightfield for each track block.
    # For example get_jump_track, get_tilt_track
    # using `virtual_track` to create non-collision mesh for collocation method in training.
    # NOTE: The heightfield is not used for building mesh in simulation, just representing the terrain
    # data relative to the block_starting_height_px in height values.
    track_trimesh, track_heightfield, block_info, height_offset_px = getattr(self, "get_" + obstacle_name + "_track")(
        wall_thickness,
        starting_trimesh,
        starting_heightfield,
        difficulty= difficulty,
        heightfield_noise= heightfield_noise[
            self.track_block_resolution[0] * (obstacle_idx + 1): self.track_block_resolution[0] * (obstacle_idx + 2)
        ] if "heightfield_noise" in locals() else None,
        virtual= virtual_track,
    )
```
* 从starting_terrain构造新的terrain

4. add_terrain_to_sim: 添加terrain to simulation
```python
self.initialize_track()
self.build_heightfield_raw(): # initialize empty terrain
self.initialize_track_info_buffer()
```

5. add_plane_to_sim: 和add_track_to_sim一样，利用add_trimesh_tosim将nparray添加到simulati
6. get_xx_depth: 根据block_infos和positions_in_block获取penetrated depths;
* 1545-1804:
```python
    def get_jump_penetration_depths(self,
            block_infos,
            positions_in_block,
            mask_only= False,
        ):
        in_up_mask = torch.logical_and(
            positions_in_block[:, 0] <= block_infos[:, 1],
            block_infos[:, 2] > 0.,
        )
        in_down_mask = torch.logical_and(
            positions_in_block[:, 0] <= block_infos[:, 1],
            block_infos[:, 2] < 0.,
        )
        jump_over_mask = torch.logical_and(
            positions_in_block[:, 2] > block_infos[:, 2],
            positions_in_block[:, 2] > 0, # to avoid the penetration of virtual obstacle in jump down.
        ) # (n_points,)
        if (block_infos[:, 2] < 0.).any():
            print("Warning: jump down is deprecated, use down instead.")

        penetrated_mask = torch.logical_and(
            torch.logical_or(in_up_mask, in_down_mask),
            (torch.logical_not(jump_over_mask)),
        )
        if mask_only:
            return penetrated_mask.to(torch.float32)
        penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
        penetrate_up_mask = torch.logical_and(penetrated_mask, in_up_mask)
        penetration_depths_buffer[penetrate_up_mask] = block_infos[penetrate_up_mask, 2] - positions_in_block[penetrate_up_mask, 2]
        penetrate_down_mask = torch.logical_and(penetrated_mask, in_down_mask)
        penetration_depths_buffer[penetrate_down_mask] = 0. - positions_in_block[penetrate_down_mask, 2]
        return penetration_depths_buffer
```
* 渗透距离仅仅计算某个轴的渗入长度
* 它这个不计算渗透体积
7. 信息访问函数
* get_goal_position:
* in_terrain_range:
* get_terrain_heights:
* available_terrain_type_names 
* get_terrain_type_names

8. draw_xxx: 可视化函数


## function
* get_penetration_depths
```python
    def get_penetration_depths(self, sample_points, mask_only= False):
        track_idx = self.get_track_idx(sample_points, clipped= False)
        track_idx_clipped = self.get_track_idx(sample_points)
        in_track_mask = (track_idx[:, 0] >= 0) \
            & (track_idx[:, 1] >= 0) \
            & (track_idx[:, 0] < self.cfg.num_rows) \
            & (track_idx[:, 1] < self.cfg.num_cols)
        forward_distance = sample_points[:, 0] - self.cfg.border_size - (track_idx[:, 0] * self.env_length) # (N,) w.r.t a track
        block_idx = torch.floor(forward_distance / self.env_block_length).to(int) # (N,) 
        block_idx[block_idx >= self.track_info_map.shape[2]] = 0.
        # positions_in_block在块中的位置
        positions_in_block = torch.stack([
            forward_distance % self.env_block_length,
            sample_points[:, 1] - self.env_origins_pyt[track_idx_clipped[:, 0], track_idx_clipped[:, 1]][:, 1],
            sample_points[:, 2] - self.block_starting_height_map[track_idx_clipped[:, 0], track_idx_clipped[:, 1], block_idx],
        ], dim= -1) # (N, 3) related to the origin of the block, not the track.
        # information of block
        block_infos = self.track_info_map[track_idx_clipped[:, 0], track_idx_clipped[:, 1], block_idx] # (N, 3)

        penetration_depths = torch.zeros_like(sample_points[:, 0]) # shape (N,)
        for obstacle_name, obstacle_id in self.track_options_id_dict.items():
            point_masks = (block_infos[:, 0] == obstacle_id) & (in_track_mask)
            if not point_masks.any(): continue
            penetration_depths[point_masks] = getattr(self, "get_" + obstacle_name + "_penetration_depths")(
                block_infos[point_masks],
                positions_in_block[point_masks],
                mask_only= mask_only,
            )
        penetration_depths[torch.logical_not(in_track_mask)] = 0.

        return penetration_depths
```
* positions_in_block: 在block中的(x,y,z)轴坐标
* 以`get_jump_penetration_depths`为例
```python
def get_jump_penetration_depths(self,
        block_infos,
        positions_in_block,
        mask_only= False,
    ):
    in_up_mask = torch.logical_and(
        positions_in_block[:, 0] <= block_infos[:, 1],
        block_infos[:, 2] > 0.,
    )
    in_down_mask = torch.logical_and(
        positions_in_block[:, 0] <= block_infos[:, 1],
        block_infos[:, 2] < 0.,
    )
    jump_over_mask = torch.logical_and(
        positions_in_block[:, 2] > block_infos[:, 2],
        positions_in_block[:, 2] > 0, # to avoid the penetration of virtual obstacle in jump down.
    ) # (n_points,)
    if (block_infos[:, 2] < 0.).any():
        print("Warning: jump down is deprecated, use down instead.")

    penetrated_mask = torch.logical_and(
        torch.logical_or(in_up_mask, in_down_mask),
        (torch.logical_not(jump_over_mask)),
    )
    if mask_only:
        return penetrated_mask.to(torch.float32)
    penetration_depths_buffer = torch.zeros_like(penetrated_mask, dtype= torch.float32)
    penetrate_up_mask = torch.logical_and(penetrated_mask, in_up_mask)
    penetration_depths_buffer[penetrate_up_mask] = block_infos[penetrate_up_mask, 2] - positions_in_block[penetrate_up_mask, 2]
    penetrate_down_mask = torch.logical_and(penetrated_mask, in_down_mask)
    penetration_depths_buffer[penetrate_down_mask] = 0. - positions_in_block[penetrate_down_mask, 2]
    return penetration_depths_buffer
```
* 这样penetration_depth代表在z轴的偏差
* 对于stairup, 计算到最近的trace的距离怎么写?
```python
def get_stairsup_penetration_depths(self,
        block_infos,
        positions_in_block,
        mask_only= False,
    ):
    stairs_lengths = block_infos[:, 1]
    stairs_heights = block_infos[:, 2]
    nearest_stair_edge_x = torch.round(positions_in_block[:, 0] / stairs_lengths) * stairs_lengths
    nearest_stair_edge_x[nearest_stair_edge_x >= self.env_block_length] -= \
        stairs_lengths[nearest_stair_edge_x >= self.env_block_length]
    nearest_stair_edge_z = torch.round(nearest_stair_edge_x / stairs_lengths) * stairs_heights
    distance_to_edge = torch.norm(
        torch.cat([positions_in_block[:, 0], positions_in_block[:, 2]], dim= -1) - 
        torch.cat([nearest_stair_edge_x, nearest_stair_edge_z], dim= -1),
        dim= -1,
    )
    if mask_only:
        return (distance_to_edge < self.track_kwargs["stairsup"].get("residual_distance", 0.05)).to(torch.float32)
    else:
        return torch.clip(
            self.track_kwargs["stairsup"].get("residual_distance", 0.05) - distance_to_edge,
            min= 0.,
        )
```
* nearest_stair_depth_x是根据当前位置计算的最近台阶的x,z
* distance计算到最近台阶的距离
* 坐标和地形的渗入距离，不是枚举坐标和地形的所有像素点的距离来计算的；
```bash
如果 mask_only 为 True，则返回一个布尔掩码，指示物体是否在允许的穿透距离内（小于 residual_distance）。
如果 mask_only 为 False，则返回穿透深度，计算为允许的穿透距离减去到边缘的距离，并确保结果不小于 0。
```
* 踩踏点远离stair's edge时penalty最小;


## related reward
```python
    def _reward_penetrate_depth(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_depths = self.terrain.get_penetration_depths(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_depths *= torch.norm(self.volume_sample_points_vel, dim= -1) + 1e-3
        return torch.sum(penetration_depths, dim= -1)

    def _reward_penetrate_volume(self):
        if not self.check_BarrierTrack_terrain(): return torch.zeros_like(self.root_states[:, 0])
        self.refresh_volume_sample_points()
        penetration_mask = self.terrain.get_penetration_mask(self.volume_sample_points.view(-1, 3)).view(self.num_envs, -1)
        penetration_mask *= torch.norm(self.volume_sample_points_vel, dim= -1) + 1e-3
        return torch.sum(penetration_mask, dim= -1)
```
