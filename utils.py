import numpy as np 
from typing import Any, Optional
from kernels import kernel
from likelihoods import likelihood

class GP((X,Y), kernel):
    """ 
    base class for Guassian process model

    :param X: input location
    :param Y: output observations
    :param kernel: gps kernel
    :param mean_funtion: mean_function (to add)
    :param Norm: set normalisation (to add)

    """

    def __init__(self, X,Y, kernel, mean_function = None, Norm = None):

        #add mean function and normalisation

        assert X.ndim == 2
        assert Y.ndim == 2
    