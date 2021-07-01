# -*- coding: utf-8 -*-
"""
@Time ： 2021/6/30 15:06
@Auth ： JsHou
@Email : 137073284@qq.com
@File ：utils.py

"""

import os


def get_root_path():
    """return project's root path"""
    root_path = os.path.dirname(os.path.abspath(__file__))
    return root_path
