from .models import Net, PlainNet, AttnPlainNet
from .lgl import LGL, KLGL, AFGN, KCAT, LifelongRehearsal
from .KTransCat import KTransCAT, AttnKTransCAT
from .sage import SAGE, LifelongSAGE
from .GCN import GCN
from .APPNP import APPNP,APP
from .MLP import MLP
from .GAT import GAT

from .layer import FeatBrd1d
from .layer import FeatTrans1d
from .layer import FeatTransKhop
from .layer import Mlp

from .ewc_loss import EWCLoss
