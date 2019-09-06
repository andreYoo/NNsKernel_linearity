# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import cifar_input
import numpy as np
import fb_resnet_model
import tensorflow as tf
import pdb
def weight_lin_eval(kernel):
    _shape = np.shape(kernel)
    if len(_shape)==3:
        kernel = np.reshape(kernel,(_shape[1],_shape[2]))
        _dim = _shape[1]
    else:
        _dim=_shape[0]
    _wtw = np.matmul(kernel, np.transpose(kernel))
    _ident_mtx = np.identity(_dim)
    _val = float(1./_dim) *np.linalg.norm(np.subtract(_wtw,_ident_mtx),ord='fro')
    return _val
    

