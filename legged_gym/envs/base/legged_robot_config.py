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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    """足式机器人环境主配置类，包含仿真环境的所有参数设置"""
    
    class env:
        """环境基本参数设置"""
        num_envs = 4096             # 并行训练的环境数量，影响训练速度和GPU内存占用
        num_observations = 235      # 观测空间维度（智能体感知的状态信息维度）
        num_privileged_obs = None   # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
                                    # 特权观测维度，特权观测是指只有在仿真中才能获得的数据（如本体移动的线速度），用于非对称训练（批评器可访问的额外信息）
        num_actions = 12            # 动作空间维度（通常对应机器人关节数量，足式机器人的关节数一般为12个）
        env_spacing = 3.            # not used with heightfields/trimeshes 
                                    # 环境实例之间的间距（米）
        send_timeouts = True        # send time out information to the algorithm
                                    # 是否向算法发送超时信息
        episode_length_s = 20       # episode length in seconds
                                    # 每个回合的最大时长（秒）

    class terrain:
        """地形生成参数设置"""
        mesh_type = 'trimesh'       # "heightfield" # none, plane, heightfield or trimesh
                                    # 地形类型：none, plane, heightfield, trimesh
        horizontal_scale = 0.1      # 水平缩放比例（米/网格）[m]
        vertical_scale = 0.005      # 垂直缩放比例（米/网格）[m]
        border_size = 25            # 地形边界大小[m]
        curriculum = True           # 是否使用课程学习（逐步增加地形难度）
        static_friction = 1.0       # 静摩擦系数
        dynamic_friction = 1.0      # 动摩擦系数
        restitution = 0.            # 恢复系数（弹性）

        # rough terrain only（只对非平坦地形）:

        # 地形测量参数
        measure_heights = True  # 是否测量地形高度

        # X轴测量点位置（机器人前方）
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        
        # Y轴测量点位置（机器人两侧）
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
       
        # 地形选择参数
        selected = False            # select a unique terrain type and pass all arguments
                                    # 是否选择单一地形类型
        terrain_kwargs = None       # Dict of arguments for selected terrain
                                    # 特定地形的参数字典
        max_init_terrain_level = 5  # starting curriculum state
                                    # 初始课程学习等级
        terrain_length = 8.         # 单个地形块长度
        terrain_width = 8.          # 单个地形块宽度
        num_rows= 10                # number of terrain rows (levels)
                                    # 地形行数（难度等级）
        num_cols = 20               # number of terrain cols (types)
                                    # 地形列数（类型变化）
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]   # 各地形类型比例
        # trimesh only:
        slope_treshold = 0.75       # slopes above this threshold will be corrected to vertical surfaces
                                    # 坡度阈值，超过此值视为垂直面

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        """机器人初始状态设置"""
        pos = [0.0, 0.0, 1.]        # 初始化位置：x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # 初始姿态（四元数表示）：x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]   # 初始化线速度：x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]   # 初始化角速度：x,y,z [rad/s]
        default_joint_angles = {    # 默认关节角度：target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        """机器人资源文件参数"""
        file = ""                               # URDF文件路径
        name = "legged_robot"                   # actor name 机器人角色名称
        foot_name = "None"                      # name of the feet bodies, used to index body state and contact force tensors
                                                # 足部刚体名称，用于接触力计算
        penalize_contacts_on = []               # 接触惩罚的刚体列表
        terminate_after_contacts_on = []        # 接触后终止的刚体列表
        disable_gravity = False                 # 是否禁用重力
        collapse_fixed_joints = True            # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
                                                # 是否合并固定连接的刚体
        fix_base_link = False                   # fixe the base of the robot
                                                # 是否固定基座链接
        default_dof_drive_mode = 3              # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
                                                # 关节驱动模式：0=无，1=位置，2=速度，3=力矩
        self_collisions = 0                     # 1 to disable, 0 to enable...bitwise filter
                                                # 自碰撞检测：0=启用，1=禁用
        replace_cylinder_with_capsule = True    # replace collision cylinders with capsules, leads to faster/more stable simulation
                                                # 用胶囊体替代圆柱体，提高稳定性
        flip_visual_attachments = True          # Some .obj meshes must be flipped from y-up to z-up
        
        # 物理属性
        density = 0.001                 # 密度[kg/m³]
        angular_damping = 0.            # 角阻尼
        linear_damping = 0.             # 线阻尼
        max_angular_velocity = 1000.    # 最大角速度
        max_linear_velocity = 1000.     # 最大线速度
        armature = 0.                   # 臂惯量
        thickness = 0.01                # 厚度

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        """物理仿真参数"""
        dt =  0.005                 # 仿真时间步长（秒），隔多长时间仿真平台计算一次
        substeps = 1                # 物理子步数
        gravity = [0., 0. ,-9.81]   # 重力向量[m/s^2]
        up_axis = 1                 # 上轴方向：0 is y, 1 is z

        class physx:
            """NVIDIA PhysX物理引擎参数"""
            num_threads = 10                    # 物理线程数
            solver_type = 1                     # 求解器类型：0: pgs, 1: tgs
            num_position_iterations = 4         # 位置迭代次数
            num_velocity_iterations = 0         # 速度迭代次数
            contact_offset = 0.01               # 接触偏移量[m]
            rest_offset = 0.0                   # 静止偏移量[m]
            bounce_threshold_velocity = 0.5     # 反弹阈值速度0.5 [m/s]
            max_depenetration_velocity = 1.0    # 最大穿透恢复速度
            max_gpu_contact_pairs = 2**23       # GPU最大接触对数 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5  # 缓冲区大小乘数
            contact_collection = 2              # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
                                                # 接触收集模式：0=从不，1=最后子步，2=所有子步

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt