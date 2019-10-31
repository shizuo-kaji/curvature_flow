#!/usr/bin/env python

import chainer.functions as F
import numpy as np
from chainer import optimizers
import functools
import chainer.links as L

optim = {
    'SGD': optimizers.MomentumSGD,
    'Momentum': optimizers.MomentumSGD,
    'AdaDelta': optimizers.AdaDelta,
    'AdaGrad': optimizers.AdaGrad,
    'Adam': optimizers.Adam,
    'AdaBound': functools.partial(optimizers.Adam, adabound=True),
    'RMSprop': optimizers.RMSprop,
    'NesterovAG': optimizers.NesterovAG,
}
try:
    from eve import Eve
    optim['Eve'] = functools.partial(Eve)
except:
    pass
try:
    from lbfgs import LBFGS
    optim['LBFGS'] = functools.partial(LBFGS, stack_size=10)
except:
    pass
    
dtypes = {
    'fp16': np.float16,
    'fp32': np.float32
}
