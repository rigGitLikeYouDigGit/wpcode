from __future__ import annotations
import typing as T

from dataclasses import dataclass

import numpy as np

from chimaera import Data

"""everything is data"""

@dataclass
class Transform(Data):
	"""data object for transform
	aka a matrix
	"""
	matrix : np.ndarray



