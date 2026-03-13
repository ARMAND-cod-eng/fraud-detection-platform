# Production Fraud Detection Platform
# Author: Armand Junior Dongmo Notue

from .features import (
    create_uid_features,
    smooth_target_encode,
    frequency_encode
)
from .model import (
    score_batch,
    find_best_threshold
)
from .monitoring import (
    ks_test,
    psi,
    get_alert_level
)

__version__ = "1.0.0"
__author__  = "Armand Junior Dongmo Notue"
