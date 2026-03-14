from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import jax
import jax.numpy as jnp
from jax import jit

from wpsim.kine.builder import SimBuilder

"""the idea is that you would supply a python file like this,
with a top-level function called BUILD(), 
which would mutate a simbuilder during compilation
"""


def BUILD(builder:SimBuilder, buildParams=None):
	"""body of this could also be pasted from inline lambdas -
	look up preceding sim values through PARAM['key'], BODY['name'], etc
	"""
	pass

	builder.var["tvec"] = jax.jit(lambda ss : ss.bufs.bodyData[
		builder.bodyIndex("humerus")].matrix * (0,0,1))

	builder.addConstraint(
		MyConstraintType(
			"elbowCn",
			builder.bodyIndex("humerus"),
			builder.bodyIndex("radius"),
			axis=(0,0,1),
			angle=45,
			strength=builder.ramp(
				"elbowRm",
				angle(builder.body("radius"), builder.body("humerus"))
			)
		)
	)

	builder.var["elbowStrength"] = builder.ramp(
				"elbowRm",
				angle(builder.body("radius"), builder.body("humerus")))

	builder.addConstraint(
		MyConstraintType(
			"elbowCn",
			builder.bodyIndex("humerus"),
			builder.bodyIndex("radius"),
			axis=(0,0,1),
			angle=45,
			strength=builder.VAR["elbowStrength"]
		)
	)
	# run vars, place var results in constraint param buffers
	sim.runvars()
	for i in sim.constraintParamVars:
		sim.constraintTypes[type].paramBuf[i.paramName][i.constIndex] = (
			sim.vars)[i.name]






