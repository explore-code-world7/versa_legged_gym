# Import terrain
* create plane: gym.add_ground
```python
    def create_sim(self):
        """ Creates simulation, terrain_lib and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        # make a difference here
```

* create terrain: gym.add_triangle_mesh
* terrain_class's role is to provides tri_meshes
* just imports it in arbitrary way

```python
    def create_sim(self):
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly

        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        self._create_stair()
        self._create_envs()

    # def _create_stair(self):
    #     StairTerrain = get_terrain_cls(self.cfg.terrain_lib.name)
    #     self.stair = StairTerrain(self.cfg.terrain_lib, self.num_envs)
    #     self.stair.add_terrain_to_sim(self.sim, self.gym)

    def _create_terrain(self):
        if getattr(self.cfg.terrain, "selected", None) is None:
            self._create_ground_plane()
        else:
            terrain_cls = self.cfg.terrain.selected
            self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs)
            self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
            
    def add_terrain_to_sim(self, sim, gym):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.vertices.shape[0]
        tm_params.nb_triangles = self.triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.

        gym.add_triangle_mesh(sim, self.vertices.flatten(), self.triangles.flatten(), tm_params)  # 接着加mesh
```


# height_sensor
* self.terrain.height_terrain_raw
```python
    def __init__(self):
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain_lib at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain_lib mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        return self.terrain.get_terrain_heights(points)
```

* get_terrain_heights如何实现?
```python
    def get_terrain_heights(self, points):
        """ Return the z coordinate of the terrain_lib where just below the given points. """
        num_robots = points.shape[0]
        points += self.cfg.border_size
        points = (points/self.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0]-2)
        py = torch.clip(py, 0, self.heightsamples.shape[1]-2)

        heights1 = self.heightsamples[px, py]
        heights2 = self.heightsamples[px+1, py]
        heights3 = self.heightsamples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(num_robots, -1) * self.cfg.vertical_scale
```

* terrain本来由trimesh-nparray生成
```python
class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

    def add_terrain_to_map(self, terrain, row, col):
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

    def selected_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain_lib",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
```


# create terrain
可以直接从高度场构造, 修改height_field_raw, 通过convert_heightfield_to_trimesh转为ver,triangles
~~2. 也可以构造isgym.sub_terrain~~

* isgym： subterrain是个包含height_field_raw的类
```python
class SubTerrain:
    def __init__(self, terrain_name="terrain_lib", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
```
* isgym通过各种地形构造函数，修改height_field_raw, 通过convert_heightfield_to_trimesh转为ver,triangles
* build terrain
```python
    self.heightsamples_float = self.generate_fractal_noise_2d(self.xSize, self.ySize, self.tot_rows, self.tot_cols, **cfg.TerrainPerlin_kwargs)
    self.heightsamples = (self.heightsamples_float * (1 / cfg.vertical_scale)).astype(np.int16)    

    @staticmethod
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        # Gradients
        angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))
        g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
        # Interpolation
        t = f(grid)
        n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
        n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
        return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1) * 0.5 + 0.5
```
