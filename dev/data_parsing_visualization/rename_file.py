#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 20:28:19 2023

@author: shiming
"""

import os

paths = (os.path.join(root, filename)
         for root, _, filenames in os.walk('./my_results')
         for filename in filenames)

for path in paths:
    # keyword replace
    newname = path.replace('MOOP_front', 'IBEA')
    if newname != path:
        os.rename(path, newname)
