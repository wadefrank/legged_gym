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

import inspect  # 导入检查模块，用于检查对象类型

class BaseConfig:
    # 作为构造函数，在创建BaseConfig子类实例时自动调用
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods).
            以递归的方式初始化所有成员类。忽略所有以'__'开头的名称（内置方法）
        """

        # 调用静态方法开始递归初始化
        self.init_member_classes(self)
    
    @staticmethod
    def init_member_classes(obj):
        """
        递归初始化对象的所有成员类
        
        Args:
            obj: 需要初始化成员类的对象实例
        """

        # iterate over all attributes names
        # 遍历对象的所有属性名称
        for key in dir(obj):
            # disregard builtin attributes
            # 忽略内置属性（原注释说忽略所有__开头，但实际只跳过了__class__）
            # if key.startswith("__"):
            if key=="__class__":
                continue

            # get the corresponding attribute object
            # 获取对应的属性对象
            var =  getattr(obj, key)

            # check if it the attribute is a class
            # 检查该属性是否是一个类（而不是实例或其他类型）
            if inspect.isclass(var):
                # instantate the class
                # 实例化这个类
                i_var = var()
                
                # set the attribute to the instance instead of the type
                # 将属性设置为实例而不是类类型
                setattr(obj, key, i_var)

                # recursively init members of the attribute
                # 递归初始化新实例的成员类
                BaseConfig.init_member_classes(i_var)