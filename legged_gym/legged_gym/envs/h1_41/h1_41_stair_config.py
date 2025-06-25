from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

# 也许可以把torso固定了
# EnvCfg
class H1_41RoughCfg(LeggedRobotCfg):

    class commands(LeggedRobotCfg.commands):
        # resampling_time = 5 # [s]
        lin_cmd_cutoff = 0.2
        ang_cmd_cutoff = 0.2
        is_goal_based = True
        class goal_based:
            # the ratios are related to the goal position in robot frame
            x_ratio = None # sample from lin_vel_x range
            y_ratio = 1.2
            yaw_ratio = 1.
            follow_cmd_cutoff = True
            x_stop_by_yaw_threshold = 1. # stop when yaw is over this threshold [rad]

    class termination:
        # additional factors that determines whether to terminates the episode
        termination_terms = [
            "roll",
            "pitch",
            "z_low",
            "z_high",
            # "out_of_track",
        ]

        roll_kwargs = dict(
            threshold= 0.8, # [rad]
            tilt_threshold= 1.5,
        )
        pitch_kwargs = dict(
            threshold= 1.6, # [rad] # for leap, jump
            jump_threshold= 1.6,
            leap_threshold= 1.5,
        )
        z_low_kwargs = dict(
            threshold= 0.15, # [m]
        )
        z_high_kwargs = dict(
            threshold= 1.5, # [m]
        )
        out_of_track_kwargs = dict(
            threshold= 1., # [m]
        )

        check_obstacle_conditioned_threshold = True
        timeout_at_border = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = "plane"
        mesh_type = None
        # selected = "PlaneTerrain"
        selected = "BarrierTrack"

        # select height
        # selected = None
        curriculum = False
        measure_heights = True

        # terrain_kwargs = {
        #
        # }

        block_width = 8
        block_length = 8
        # 网格的横纵尺寸
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.05  # [m]
        step_width = 0.5
        step_height = 0.25
        slope_threshold = 0.75

        terrain_width = block_width
        terrain_length = block_length
        num_rows = 1
        num_cols = 1

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

        add_noise = True
        noise_limit = 0.1

        border_size = 2.0

        #这玩意
        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )

        BarrierTrack_kwargs = dict(
            options= [
                # "jump",
                # "leap",
                # "hurdle",
                # "down",
                # "tilted_ramp",
                "stairsup",
                # "stairsdown",
                # "discrete_rect",
                # "slope",
                # "wave",
            ], # each race track will permute all the options
            jump= dict(
                height= [0.05, 0.5],
                depth= [0.1, 0.3],
                # fake_offset= 0.1,
            ),
            leap= dict(
                length= [0.05, 0.8],
                depth= [0.5, 0.8],
                height= 0.2, # expected leap height over the gap
                # fake_offset= 0.1,
            ),
            hurdle= dict(
                height= [0.05, 0.5],
                depth= [0.2, 0.5],
                # fake_offset= 0.1,
                curved_top_rate= 0.1,
            ),
            down= dict(
                height= [0.1, 0.6],
                depth= [0.3, 0.5],
            ),
            tilted_ramp= dict(
                tilt_angle= [0.2, 0.5],
                switch_spacing= 0.,
                spacing_curriculum= False,
                overlap_size= 0.2,
                depth= [-0.1, 0.1],
                length= [0.6, 1.2],
            ),
            slope= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-3.14, 0, 1.57, -1.57],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopeup= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            slopedown= dict(
                slope_angle= [0.2, 0.42],
                length= [1.2, 2.2],
                use_mean_height_offset= True,
                face_angle= [-0.2, 0.2],
                no_perlin_rate= 0.2,
                length_curriculum= True,
            ),
            stairsup= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                residual_distance= 0.05,
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            stairsdown= dict(
                height= [0.1, 0.3],
                length= [0.3, 0.5],
                num_steps= [3, 19],
                num_steps_curriculum= True,
            ),
            discrete_rect= dict(
                max_height= [0.05, 0.2],
                max_size= 0.6,
                min_size= 0.2,
                num_rects= 10,
            ),
            wave= dict(
                amplitude= [0.1, 0.15], # in meter
                frequency= [0.6, 1.0], # in 1/meter
            ),
            track_width= 3.2,
            track_block_length= 2.4,
            wall_thickness= (0.01, 0.6),
            wall_height= [-0.5, 2.0],
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 0.8,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
            randomize_obstacle_order= True,
            n_obstacles_per_track= 1,
        )

    class env(LeggedRobotCfg.env):
        # 3 + 3 + 3 + 12 + 12 + 12 + 2 = 47
        # 11 + 41*3 = 134
        # 这些都不用定义了
        # num_observations = 132
        # num_privileged_obs = 135
        # num_actions = 41
        obs_components = [
            # "lin_vel",      # 3
            "ang_vel",      # 3
            "projected_gravity",    #2
            "commands",     # 3
            "dof_pos",      # 41
            "dof_vel",      # 41
            "last_actions", # 41
            "height_measurements",
            # "forward_depth",
        ]
        privileged_obs_components = [
            "lin_vel",      # 3
            "ang_vel",  # 3
            "projected_gravity",  # 2
            "commands",  # 3
            "dof_pos",  # 41
            "dof_vel",  # 41
            "last_actions",  # 41
            "height_measurements",
        ]
        num_envs = 2048

    class noise(LeggedRobotCfg.noise):
        add_noise=False

        # class noise_scales(LeggedRobotCfg.noise.noise_scales):
        #     forward_depth = 0.01

    # camera configs
    # class sensor(LeggedRobotCfg):
    #     class forward_camera:
    #         obs_components = ["forward_depth"]
    #         resolution = [int(480/4), int(640/4)]
    #         position = [0.0, 0.0, 0.50]
    #         rotation = [0.0, 0.0, 0.0]
    #         resized_resolution = [48, 64]
    #         output_resolution = [48, 64]
    #         horizontal_fov = [86, 90]
    #         crop_top_bottom = [int(48/4), 0]
    #         crop_left_right = [int(28/4), int(36/4)]
    #         near_plane = 0.05
    #         depth_range = [0.0, 3.0]
    #
    #         latency_range = [0.08, 0.142]
    #         latency_resampling_time = 5.0
    #         refresh_duration = 1/10 # [s]
    #
    #     class proprioception:
    #         obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"]
    #         latency_range = [0.005, 0.045] # [s]
    #         latency_resampling_time = 5.0 # [s]

    class viewer(LeggedRobotCfg.viewer):
        debug_viz = True
        draw_measure_heights = True
        draw_sensors = True
        draw_volume_sample_points = True
        pos = [-1., 0., 1.0]
        lookat = [0., 0., 0.3]

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
        decimation = 4

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025
        no_camera = True

        # # draw body points
        body_measure_points = { # transform are related to body frame
            "ankle_roll": dict(
                x= [i for i in np.arange(-0.09, 0.23, 0.04)],
                y= [i for i in np.arange(-0.05, 0.05, 0.02)],
                z= [i for i in np.arange(-0.05, 0.01, 0.015)],
                transform= [0., 0., 0.005, 0., 0., 0.],
            ),
        }


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

            # go1_rough
            tracking_lin_vel = 1.
            tracking_ang_vel = 1.
            # what？
            energy_substeps = -2e-7     # 对human要降
            stand_still = -1.
            # peculiar joints' error
            # dof_error_named = -1.
            dof_error = -0.005
            collision = -0.05
            lazy_stop = -3.
            # penalty for hardware safety
            exceed_dof_pos_limits = -0.4
            exceed_torque_limits_l1norm = -0.4
            dof_vel_limits = -0.4
            penetrate_depth = -0.005

            # h1_origin
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
            lin_vel_z = -0.2
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-8
            dof_vel = -1e-5
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -1e-3
            dof_pos_limits = -5.0
            alive = 1.5
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18
            torques = 1e-6
            # hand_sup = -100.0

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

        encoder_component_names = ["height_measurements"]
        encoder_class_name = "MlpModel"
        class encoder_kwargs:
            hidden_sizes = [128, 64]
            nonlinearity = "CELU"
        encoder_output_size = 32
        critic_encoder_component_names = ["height_measurements"]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 1e-2

    class runner(LeggedRobotCfgPPO.runner):
        # policy_class_name = "ActorCritic"
        # policy_class_name = "ActorCriticRecurrent"
        policy_class_name = "EncoderActorCriticRecurrent"
        # policy_class_name = "TriActCritic"
        algorithm_class_name = "PPO"
        max_iterations = 1000
        run_name = ''
        experiment_name = 'h1_41'
