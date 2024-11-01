from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


def grayCode(n=4):
	""" generates full cycle of a Gray code,
	of 'bit depth' n
	currently outputs an array of length 2 ^ n
	copied from geeksforgeeks, by Mohit Kumar
	and Ravi Chandra Enaganti """

	code = [ "0", "1" ] # basic pattern
	i = 2

	# i << n
	# is equal to
	# i * ( 2 ^ n )
	# superfast bit-shifting maths

	while i < 1 << n:
		for j in range( i - 1, -1, -1):
			""" iterate through existing code in reverse
			this is the 'reflection' in the original patent """
			code.append( code[j] )

		# append 0 to the first half
		for j in range(i):
			code[j] = "0" + code[j]

		# append 1 to the second half
		for j in range(i, 2 * i ):
			code[j] = "1" + code[j]

		i = i << 1
	return code


