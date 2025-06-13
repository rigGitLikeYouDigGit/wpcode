from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


""" can we use tensorflow to emulate a simple IK system
following a target point?


jointA ----> joint B ----> point P


"""

import tensorflow as tf
print(tf, tf.__file__, tf.__dict__, tf.__version__)
print(tf.keras)

@tf.function
def mat4x4c(xAxis=tf.constant([1.0, 0.0, 0.0]),
           yAxis=tf.constant((0.0, 1.0, 0.0)),
           zAxis=tf.constant((0.0, 0.0, 1.0)),
           pos=tf.constant((0.0, 0.0, 0.0))
           )->tf.Tensor:
	return tf.constant([xAxis, yAxis, zAxis, pos])


