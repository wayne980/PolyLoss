# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 19:16:47 2019

@author: FMC_417_1
"""

from vocab import Vocabulary
import visualscan
visualscan.evalrank("./runs/coco/ckpt1/model_best.pth.tar", data_path="/data4/weijiwei/mtfn", split="test",fold5=False)
