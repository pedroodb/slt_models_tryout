from .hyperparameters import HyperParameters

from .rwth import hp as rwth
from .gsl import hp as gsl
from .lsat import hp as lsat

HP_DICT = {"RWTH_PHOENIX_2014T": rwth, "GSL": gsl, "LSAT": lsat}
