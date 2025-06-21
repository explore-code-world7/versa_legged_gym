import numpy as np
import torch
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym.terrain_utils import *


# import matplotlib.pyplot as plt

class StairTerrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_envs) -> None:
        self.cfg = cfg
        # self.num_robots = num_robots
        self.type = cfg.mesh_type
        # if self.type in ["none", 'plane']:
        #     return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        self.slope_threshold = cfg.slope_threshold

        self.envs_per_row = cfg.num_rows
        self.envs_per_col = cfg.num_cols

        self.step_width = cfg.step_width
        self.step_height = cfg.step_height

        # self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]
        #
        # self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        #
        # self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        # self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        #
        # self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        # self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        # self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border
        #
        # self.slope_threshold = cfg.slope_threshold

        # 把terrain的参数，放到terrain_kwargs中
        self.block_width = cfg.block_width
        self.block_length = cfg.block_length
        self.stair_height = int((self.block_width/(self.step_width/self.horizontal_scale))*(self.step_height/self.vertical_scale))

        self.heightsamples = self.generate_stair_terrain()
        self.heightfield_raw_pyt = torch.tensor(self.heightsamples, device= "cpu")

        # if self.type=="trimesh":
            # transform trimesh into real terrains
            # why does slope_threshold loses effect when concatenate two planes with different height
        self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.heightsamples,
                                                                                        self.horizontal_scale,
                                                                                        self.vertical_scale,
                                                                                        self.slope_threshold)

        # # 输出真实高度, 和terrain_height
        # print(f"height_field_raw_unique = {np.unique(self.height_field_raw)}")
        # print(f"stair_height= {self.stair_height}")
        # print("#"*50+"see miracle height")
        # print(np.unique(self.height_field_raw)*self.horizontal_scale)
        # print(f"[0,1]: {np.unique(self.height_field_raw[:block_rows, block_cols:2*block_cols])}")
        # if self.vertices is not None:
        #     print(f"self.vertices = {np.unique(self.vertices[..., 2])}")
        #     # vertices: [x, y, z]

    def generate_stair_terrain(self):
        # Return : pixel height of terrain
        print(self.block_width)
        print(self.horizontal_scale)
        self.block_rows = int(self.block_width/self.horizontal_scale)
        self.block_cols = int(self.block_length/self.horizontal_scale)
        # print(self.envs_per_row, self.block_rows)
        height_field = np.zeros((3*self.envs_per_row*self.block_rows, 2*self.envs_per_col*self.block_cols))

        # 为什么height_field_raw这么大?
        for i in range(self.envs_per_row):
            for j in range(self.envs_per_col):
                base_rows = i*3*self.block_rows
                base_cols = j*2*self.block_cols

                # configue real width and length of block, with width and length of stair following
                height_field[base_rows+self.block_rows:base_rows+2*self.block_rows, base_cols:base_cols+self.block_cols] = self.generate_stair(self.step_width, self.step_height)

                stair_scale_height = np.max(height_field[base_rows+self.block_rows:base_rows+2*self.block_rows, base_cols:base_cols+self.block_cols])
                # print(f"stair_height = {stair_scale_height}")

                height_field[base_rows+2*self.block_rows:base_rows+3*self.block_rows, base_cols:2*base_cols+self.block_cols] += stair_scale_height
                # height_field[base_rows+2*self.block_rows:base_rows+3*self.block_rows, base_cols+self.block_cols:base_cols+2*self.block_cols] += stair_scale_height

                height_field[base_rows+self.block_rows:base_rows+2*self.block_rows, base_cols+self.block_cols:base_cols+2*self.block_cols] = self.generate_stair(self.step_width, - self.step_height)+2*stair_scale_height

                height_field[base_rows:base_rows+self.block_rows, base_cols+self.block_cols:base_cols+2*self.block_cols] += 2*stair_scale_height

        return height_field


    # @staticmethod
    def generate_stair(self, step_width, step_height):

        height_field = np.zeros((self.block_rows, self.block_cols))

        _step_width = int(step_width / self.horizontal_scale)
        # 0.5/0.05=10
        _step_height = int(step_height / self.vertical_scale)

        num_steps = int(self.env_width / step_width)
        # 120/10=12;
        height = _step_height  # step height在前面已经改变

        # 总共 12*0.25 = 3;
        # 对应的scale的级数 = 3/0.05=60
        for i in range(num_steps):
            height_field[i * _step_width: (i + 1) * _step_width, :] += height
            height += _step_height

        # import pdb; pdb.set_trace()
        return height_field


    def add_terrain_to_sim(self, gym, sim, device="cpu"):
        self.gym = gym
        self.sim = sim
        self.device = device
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.vertices.shape[0]
        tm_params.nb_triangles = self.triangles.shape[0]
        tm_params.transform.p.x = -1.
        tm_params.transform.p.y = -1.
        tm_params.transform.p.z = 0.
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        # num_envs = self.num_envs
        # envs_per_row = int(sqrt(num_envs))
        # envs_per_col = num_envs//envs_per_row

        # self.vertices 表全局坐标系
        # self.triangles 表对应的三角形

        # for i in range(envs_per_row):
        #     for j in range(envs_per_col):
        #         vertices2 = self.vertices.copy()
        #         vertices2[:,0] += i*self.env_width
        #         vertices2[:,1] += j*self.env_length

        self.gym.add_triangle_mesh(
            self.sim, self.vertices.flatten(), self.triangles.flatten(), tm_params)  # 接着加mesh

        # add env origins
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))
        for row_idx in range(self.cfg.num_rows):
            for col_idx in range(self.cfg.num_cols):
                print("env_width", self.env_width, self.env_length)
                origin_x = (row_idx + 0.5) * self.env_length
                origin_y = (col_idx + 0.5) * self.env_width
                self.env_origins[row_idx, col_idx] = [
                    origin_x,
                    origin_y,
                    self.heightsamples[
                        int(origin_x / self.cfg.horizontal_scale),
                        int(origin_y / self.cfg.horizontal_scale),
                    ] * self.cfg.vertical_scale,
                ]
        self.heightfield_raw_pyt = torch.from_numpy(self.heightsamples).to(device= self.device).float()


    def elevation_terrain(self, terrain, elevate_height):
        terrain.height_field_raw[:,:] += elevate_height
        return terrain

    @torch.no_grad()
    def get_terrain_heights(self, points):
        """ Get the terrain_lib heights below the given points """
        points_shape = points.shape
        points = points.view(-1, 3)
        points_x_px = (points[:, 0] / self.cfg.horizontal_scale).to(int)
        points_y_px = (points[:, 1] / self.cfg.horizontal_scale).to(int)
        out_of_range_mask = torch.logical_or(
            torch.logical_or(points_x_px < 0, points_x_px >= self.heightfield_raw_pyt.shape[0]),
            torch.logical_or(points_y_px < 0, points_y_px >= self.heightfield_raw_pyt.shape[1]),
        )
        points_x_px = torch.clip(points_x_px, 0, self.heightfield_raw_pyt.shape[0] - 1)
        points_y_px = torch.clip(points_y_px, 0, self.heightfield_raw_pyt.shape[1] - 1)
        heights = self.heightfield_raw_pyt[points_x_px, points_y_px] * self.cfg.vertical_scale
        heights[out_of_range_mask] = - torch.inf
        heights = heights.view(points_shape[:-1])
        return heights


    @torch.no_grad()
    def get_terrain_heights(self, points):
        """ Get the terrain heights below the given points """
        points_shape = points.shape
        points = points.view(-1, 3)
        points_x_px = (points[:, 0] / self.cfg.horizontal_scale).to(int)
        points_y_px = (points[:, 1] / self.cfg.horizontal_scale).to(int)
        out_of_range_mask = torch.logical_or(
            torch.logical_or(points_x_px < 0, points_x_px >= self.heightfield_raw_pyt.shape[0]),
            torch.logical_or(points_y_px < 0, points_y_px >= self.heightfield_raw_pyt.shape[1]),
        )
        points_x_px = torch.clip(points_x_px, 0, self.heightfield_raw_pyt.shape[0] - 1)
        points_y_px = torch.clip(points_y_px, 0, self.heightfield_raw_pyt.shape[1] - 1)
        heights = self.heightfield_raw_pyt[points_x_px, points_y_px] * self.cfg.vertical_scale
        heights[out_of_range_mask] = - torch.inf
        heights = heights.view(points_shape[:-1])
        return heights

# 如何获取全局坐标系下的高度
# 1. if height_field_raw是全局坐标，直接按索引即可；
# 2. if height_field_raw是环境下的坐标, 那self.vertices赋值后直接add_trimesh_to_sim错误，不可能


    # @torch.no_grad()
    # def get_terrain_heights(self, points):
    #     """ Get the terrain_lib heights below the given points """
    #     points_shape = points.shape
    #     points = points.view(-1, 3)
    #     points_x_px = (points[:, 0] / self.cfg.horizontal_scale).to(int)
    #     points_y_px = (points[:, 1] / self.cfg.horizontal_scale).to(int)
    #     out_of_range_mask = torch.logical_or(
    #         torch.logical_or(points_x_px < 0, points_x_px >= self.heightfield_raw_pyt.shape[0]),
    #         torch.logical_or(points_y_px < 0, points_y_px >= self.heightfield_raw_pyt.shape[1]),
    #     )
    #     points_x_px = torch.clip(points_x_px, 0, self.heightfield_raw_pyt.shape[0] - 1)
    #     points_y_px = torch.clip(points_y_px, 0, self.heightfield_raw_pyt.shape[1] - 1)
    #     heights = self.heightfield_raw_pyt[points_x_px, points_y_px] * self.cfg.vertical_scale
    #     heights[out_of_range_mask] = - torch.inf
    #     heights = heights.view(points_shape[:-1])
    #     return heights
