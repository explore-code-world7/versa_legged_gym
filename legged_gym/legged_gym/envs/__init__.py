# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO, A1PlaneCfg, A1RoughCfgTPPO
from .base.legged_robot import LeggedRobot
from .base.legged_robot_field import LeggedRobotField
from .base.robot_field_noisy import RobotFieldNoisy
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_field_config import A1FieldCfg, A1FieldCfgPPO
from .a1.a1_field_distill_config import A1FieldDistillCfg, A1FieldDistillCfgPPO
from .go1.go1_field_config import Go1FieldCfg, Go1FieldCfgPPO
from .go1.go1_field_distill_config import Go1FieldDistillCfg, Go1FieldDistillCfgPPO
from .go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO
from .go2.go2_field_config import Go2FieldCfg, Go2FieldCfgPPO
from .go2.go2_distill_config import Go2DistillCfg, Go2DistillCfgPPO


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "go1_field", LeggedRobot, Go1FieldCfg(), Go1FieldCfgPPO())
task_registry.register( "go1_distill", LeggedRobot, Go1FieldDistillCfg(), Go1FieldDistillCfgPPO())
task_registry.register( "go2", LeggedRobot, Go2RoughCfg(), Go2RoughCfgPPO() )
task_registry.register( "go2_field", RobotFieldNoisy, Go2FieldCfg(), Go2FieldCfgPPO() )
task_registry.register( "go2_distill", RobotFieldNoisy, Go2DistillCfg(), Go2DistillCfgPPO() )


## The following tasks are for the convinience of opensource
from .a1.a1_remote_config import A1RemoteCfg, A1RemoteCfgPPO
task_registry.register( "a1_remote", LeggedRobot, A1RemoteCfg(), A1RemoteCfgPPO() )
from .go1.go1_remote_config import Go1RemoteCfg, Go1RemoteCfgPPO
task_registry.register( "go1_remote", LeggedRobot, Go1RemoteCfg(), Go1RemoteCfgPPO() )



from legged_gym.envs.h1_41.h1_41_env import H1_41Robot
from legged_gym.envs.h1_41.h1_field_env import H1_Field_Robot
from legged_gym.envs.h1_41.h1_12_stair_config import H1_12_StairCfg, H1_12_StairCfgPPO

from legged_gym.envs.h1_41.h1_41_config import H1_41RoughCfg, H1_41RoughCfgPPO
from legged_gym.envs.h1_41.h1_41_hf_config import H1_41_HF_RoughCfg, H1_41_HF_RoughCfgPPO
from legged_gym.envs.h1_41.h1_45_hf_config import H1_45_HF_RoughCfg, H1_45_HF_RoughCfgPPO
from legged_gym.envs.h1_41.h1_51_hf_config import H1_51_HF_RoughCfg, H1_51_HF_RoughCfgPPO

task_registry.register( "h1_41", H1_41Robot, H1_41RoughCfg(), H1_41RoughCfgPPO())
task_registry.register( "h1_41_hf", H1_41Robot, H1_41_HF_RoughCfg(), H1_41_HF_RoughCfgPPO())
task_registry.register( "h1_45_hf", H1_41Robot, H1_45_HF_RoughCfg(), H1_45_HF_RoughCfgPPO())
task_registry.register( "h1_51_hf", H1_41Robot, H1_51_HF_RoughCfg(), H1_51_HF_RoughCfgPPO())

from legged_gym.envs.h1_41.h1_41_stair_config import H1_41_StairCfg, H1_41_StairCfgPPO
from legged_gym.envs.h1_41.h1_45_stair_config import H1_45_StairCfg, H1_45_StairCfgPPO
from legged_gym.envs.h1_41.h1_51_stair_config import H1_51_StairCfg, H1_51_StairCfgPPO

task_registry.register( "h1_12_stair", H1_Field_Robot, H1_12_StairCfg(), H1_12_StairCfgPPO())
task_registry.register( "h1_41_stair", H1_Field_Robot, H1_41_StairCfg(), H1_41_StairCfgPPO())
task_registry.register( "h1_45_stair", H1_Field_Robot, H1_45_StairCfg(), H1_45_StairCfgPPO())
task_registry.register( "h1_51_stair", H1_Field_Robot, H1_51_StairCfg(), H1_51_StairCfgPPO())