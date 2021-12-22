from models.LiESN import get_default, LiESNRegressorFitter

from rc_chaos.Methods.Models.esn.esn_rc_dyst_copy import esn
from rc_chaos.Methods.RUN import new_args_dict

models = {'LiESN': LiESNRegressorFitter,
          'esn': esn}
