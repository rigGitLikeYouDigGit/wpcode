import math
import time
import matplotlib.pyplot as plt
import numpy as np

import jax
from flax import nnx

from jax.scipy.spatial import transform
import jax.numpy as jnp

#from wpm import cmds, oma

[
	"C:/Users/arthu/AppData/Local/Programs/Python/Python311/DLLs",
"C:/Users/arthu/AppData/Local/Programs/Python/Python311/Lib",
"C:/Users/arthu/AppData/Local/Programs/Python/Python311",
"C:/Users/arthu/AppData/Local/Programs/Python/Python311/Lib/site-packages",
	"C:/Users/arthu/Documents/code/wpcode/python"
]

jnp.set_printoptions(suppress=True)

def newTfMat(
		x=(1,0,0),
		y=(0,1,0),
		z=(0,0,1),
		t=(0,0,0)
):
	"""return new 4x3 matrix for a transform?"""
	return jnp.array([
		x,
		y,
		z,
		#t
	])


def forward(params, system):
	"""apply parametres to system"""
	# return transform.Rotation.from_euler("xyz", params, True) @ system["joints"][0]
	return transform.Rotation.from_euler("xyz", params, False).as_matrix() @ system["joints"][0]

def lossFn(params, system):
	"""don't include loss here?
	check how brax does it"""
	return jnp.linalg.norm(
		forward(params, system) * jnp.array([2.0, 0, 0]) - system["target"]
	)


@jax.jit
def runFn(params, system):

	grads = jax.grad(lossFn)(params, system)
	return grads



def rebuildOriginAim():
	"""build joints and target
	a single pivot at origin tries to match an end effector to a moving target
	"""

	system = {
		"joints" : [
			newTfMat()
		],
		#"target" : jnp.array([3.0, 0.0, 0.0])
		"target" : np.array([3.0, 0.0, 0.0])
	}
	#joint = transform.Rotation.from_euler("x", 0.0, True)
	#params = jnp.array([0.0, 0.0, 0.0])
	params = np.array([0.0, 0.0, 0.0])

	nSteps = 100
	#times = jnp.zeros(100)
	times = np.zeros(nSteps)

	angles = np.zeros((nSteps, 3))

	losses = np.zeros(nSteps)

	for i in range(nSteps):
		t = math.sin(i * 0.1)
		#print("t", t)
		#system["target"].at[1].set(t)
		system["target"][1] = t
		#times.at[i].set(t)
		times[i] = t
		losses[i] = lossFn(params, system)
		grads = runFn(params, system) * lossFn(params, system) * 0.9

		params -= grads# * 0.5
		#params += grads #* 0.5
		system["joints"][0] = forward(params, system)
		angles[i] = params
		#print(system)
		#print(params)
		#time.sleep(0.5)

	p = plt.plot(times)
	plt.plot(angles)
	plt.plot(losses)
	#print(times)
	plt.show()






if __name__ == '__main__':

	rebuildOriginAim()
	import sys
	# for i in sys.path:
	# 	print(i)


