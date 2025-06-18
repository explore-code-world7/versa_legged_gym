from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


# 也许可以把torso固定了
# EnvCfg
class H1_41RoughCfg(LeggedRobotCfg):
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "plane"
        # measure_heights = True
        num_rows = 2
        num_cols = 2

        mesh_type = None
        selected = "StairTerrain"
        block_width = 6.
        block_length = 6.
        # 网格的横纵尺寸
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.05  # [m]
        step_width = 0.5
        step_height = 0.25
        slope_threshold = 1000

        static_friction = 0.8
        dynamic_friction = 0.8
        restitution = 0.5

        # mesh_type = "trimesh" # Don't change
        # num_rows = 20
        # num_cols = 50
        # selected = "TerrainPerlin" # "BarrierTrack" or "TerrainPerlin", "TerrainPerlin" can be used for training a walk policy.
        # max_init_terrain_level = 0
        # border_size = 5
        # slope_treshold = 20.
        #
        # curriculum = False # for walk
        # horizontal_scale = 0.025 # [m]
        # # vertical_scale = 1. # [m] does not change the value in hightfield
        # pad_unavailable_info = True
        # TerrainPerlin_kwargs = dict(
        #     zScale= [0.08, 0.15],
        #     # zScale= 0.15, # Use a constant zScale for training a walk policy
        #     frequency= 10,
        # )

    class noise(LeggedRobotCfg.noise):
        add_noise=False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            # 12+1=13
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,

            # contribute to revolution of torso arount pelvis
            'torso_joint': 0,

            # 14
            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_pitch_joint': 0.3,
            'left_elbow_roll_joint': 0,
            'left_wrist_pitch_joint': 0,
            'left_wrist_yaw_joint': 0,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_pitch_joint': 0.3,
            'right_elbow_roll_joint': 0,
            'right_wrist_pitch_joint': 0,
            'right_wrist_yaw_joint': 0,

            # # 14
            'L_base_link_joint': 0.0,
            # 'L_thumb_proximal_yaw_joint': 0,
            # 'L_thumb_proximal_pitch_joint': 0,
            # 'L_thumb_intermediate_joint': 0,
            # 'L_thumb_distal_joint': 0,
            'L_index_proximal_joint': 0,
            'L_index_intermediate_joint': 0,
            'L_middle_proximal_joint': 0,
            'L_middle_intermediate_joint': 0,
            'L_ring_proximal_joint': 0,
            'L_ring_intermediate_joint': 0,
            # 'L_pinky_proximal_joint': 0,
            # 'L_pinky_intermediate_joint': 0,

            'R_base_link_joint': 0.0,
            # 'R_thumb_proximal_yaw_joint': 0,
            # 'R_thumb_proximal_pitch_joint': 0,
            # 'R_thumb_intermediate_joint': 0,
            # 'R_thumb_distal_joint': 0,
            'R_index_proximal_joint': 0,
            'R_index_intermediate_joint': 0,
            'R_middle_proximal_joint': 0,
            'R_middle_intermediate_joint': 0,
            'R_ring_proximal_joint': 0,
            'R_ring_intermediate_joint': 0,
            # 'R_pinky_proximal_joint': 0,
            # 'R_pinky_intermediate_joint': 0,
        }

    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 12 + 12 + 12 + 2 = 47
        # 11 + 41*3 = 134
        num_observations = 132
        num_privileged_obs = 135
        num_actions = 41
        obs_components = [
            # "lin_vel",      # 3
            "ang_vel",      # 3
            "projected_gravity",    #2
            "commands",     # 3
            "dof_pos",      # 41
            "dof_vel",      # 41
            "last_actions", # 41
        ]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters:
        stiffness = {
            # leg
            'hip_yaw_joint': 200.,
            'hip_roll_joint': 200.,
            'hip_pitch_joint': 200.,
            'knee_joint': 300.,
            'ankle_pitch_joint': 40.,
            'ankle_roll_joint': 40.,

            # arm
            'shoulder': 200,  # 2*3
            'elbow': 200,  # 2*1
            'wrist': 200,  # 2*3

            # hand
            'base_link_joint': 200,
            # 'thumb': 40,
            'index': 40,
            'middle': 40,
            'ring': 40,
            # 'pinky': 100,

            # torso
            'torso_joint': 300,
        }  # [N*m/rad]
        damping = {
            'hip_yaw_joint': 2.5,
            'hip_roll_joint': 2.5,
            'hip_pitch_joint': 2.5,
            'knee_joint': 4,
            'ankle_pitch_joint': 2.0,
            'ankle_roll_joint': 2.0,

            # arm
            'shoulder': 4,  # 2*3
            'elbow': 4,  # 2*1
            'wrist': 2.5,  # 2*3

            # hand
            'base_link_joint': 2.5,
            # 'thumb': 2.0,
            'index': 2.0,
            'middle': 2.0,
            'ring': 2.0,
            # 'pinky': 2.0,

            # torso
            'torso_joint': 4.0,
        }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
        # initial attributes
        max_push_vel_ang = 0.
        init_dof_pos_ratio_range = [0.5, 1.5]
        init_base_vel_range = [-1., 1.]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_41dof.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        arm_name = ["elbow_pitch", "wrist_yaw", "base_link_joint"]
        finger_name = ["index", "middle", "ring", ]

        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0  # 1 to disable, 0 to enable...Wbitwise filter
        flip_visual_attachments = False
        armature = 1e-3

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1.0
        # tracking_sigma = 0.25
        # # 也不知道是不是调节成4.0更好
        # active_sigma = 1.0
        # hand_sigma

        class scales(LeggedRobotCfg.rewards.scales):
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -1.0
            # base_height = -10.0
            # dof_acc = -2.5e-7
            # dof_vel = -1e-3
            # feet_air_time = 0.0
            # collision = 0.0
            # action_rate = -0.01
            # dof_pos_limits = -5.0
            # alive = 0.15
            # hip_pos = -1.0
            # contact_no_vel = -0.2
            # feet_swing_height = -20.0
            # contact = 0.18

            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-5
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 1.5
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 1.8
            hand_sup = -100.0

# RunnerCfg
class H1_41RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [400, 200, 100]
        critic_hidden_dims = [400, 200, 100]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size =  256
        rnn_num_layers = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 1e-2
    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = "ActorCritic"
        policy_class_name = "ActorCriticRecurrent"
        # policy_class_name = "TriActCritic"
        algorithm_class_name = "PPO"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'h1_41'