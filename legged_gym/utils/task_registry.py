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

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    """任务注册表类，用于管理强化学习环境的注册和创建
    
    该类作为中心化的注册中心，维护了任务名称到具体环境类及配置的映射关系，提供了统一的 环境创建 和 算法运行器创建 的接口
    
    不同任务对应不同的机器人
    """


    def __init__(self):
        """初始化空的注册表"""
        self.task_classes = {}  # 存储任务名称到环境类的映射
        self.env_cfgs = {}      # 存储任务名称到环境配置的映射
        self.train_cfgs = {}    # 存储任务名称到训练配置的映射
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        """注册一个新的任务到注册表中
        
        Args:
            name (str): 任务的唯一标识名称
            task_class (VecEnv): 向量化环境类
            env_cfg (LeggedRobotCfg): 环境配置对象
            train_cfg (LeggedRobotCfgPPO): 训练配置对象
        """
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        """根据名称获取注册的环境类"""
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """获取指定任务的配置对象
        
        Args:
            name (str): 任务名称
            
        Returns:
            Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]: 环境配置和训练配置的元组
        """
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed

        # 确保环境配置和训练配置使用相同的随机种子
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.
            根据注册的任务名称或提供的配置文件创建用于训练强化学习的环境实例，支持命令行参数覆盖配置
            
        Args:
            name (string): Name of a registered env. （注册环境的任务名称）
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None. （Isaac Gym命令行参数，为None时自动获取）
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.（用于覆盖注册配置的环境配置）

        Raises:
            ValueError: Error if no registered env corresponds to 'name' （当没有对应名称的注册任务时抛出）

        Returns:
            isaacgym.VecTaskPython: The created environment （创建的环境实例）
            Dict: the corresponding config file （对应的配置对象）
        """
        # if no args passed get command line arguments
        # 如果没有提供args参数，则获取命令行参数
        if args is None:
            args = get_args()

        # check if there is a registered env with that name
        # 检查是否存在对应名称的注册任务
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        
        # 如果没有提供环境配置，则从注册表中加载
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        # 使用命令行参数覆盖配置（如果有具体的）
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        
        # 设置随机种子以确保实验可重复性
        set_seed(env_cfg.seed)

        # parse sim params (convert to dict first) 
        # 解析仿真参数（首先转换为字典）
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)

        # 创建环境实例
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.
            创建用于训练强化学习算法的运行器，支持从 注册的名字 或者 提供的配置文件 创建
        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)（要训练的环境（未来计划从算法中移除））
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.（注册的任务名称，为None时使用提供的配置文件，默认为None）
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.（Isaac Gym命令行参数，如果是None，get_args()会被取消，默认为None）
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.（训练配置文件）
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). （Tensorboard日志根目录，设为None不记录日志）
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided（当既没有提供name也没有提供train_cfg时抛出）
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored（当同时提供name和train_cfg时忽略name参数）

        Returns:
            PPO: The created algorithm（创建的算法运行器）
            Dict: the corresponding config file（对应的配置文件）
        """
        # if no args passed get command line arguments
        # 如果获取命令行参数
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        # 创建文件夹
        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # 创建算法运行器实例
        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        # 恢复训练逻辑：在创建新日志目录前保存恢复路径
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            # 加载之前训练的模型
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
# 创建全局任务注册表实例
task_registry = TaskRegistry()