

from __future__ import annotations
import typing as T



class Counter(int):
	def __new__(cls, start:int):
		return  int.__new__(cls, start)
	def inc(self)->int:
		self += 1
		return self


if __name__ == '__main__':

	c = Counter(3)
	print(c, type(c))

	print(c.inc(), c)
	print(c.inc(), c)



