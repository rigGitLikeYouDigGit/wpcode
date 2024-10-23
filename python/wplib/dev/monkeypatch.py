
from __future__ import annotations
import typing as T

"""a very beautiful thing by bricef on github - adding new methods to
builtin types
maybe we can replace the proxy madness
this of course is way more sensible

didn't move forwards with this since it means modifying every single instance
of a builtin in the interpreter
"""

# found this from Armin R. on Twitter, what a beautiful gem ;)

import ctypes
from types import MappingProxyType, MethodType

# figure out side of _Py_ssize_t
if hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    _Py_ssize_t = ctypes.c_int64
else:
    _Py_ssize_t = ctypes.c_int

# regular python
class _PyObject(ctypes.Structure):
    pass
_PyObject._fields_ = [
    ('ob_refcnt', _Py_ssize_t),
    ('ob_type', ctypes.POINTER(_PyObject))
]

# python with trace
if object.__basicsize__ != ctypes.sizeof(_PyObject):
    class _PyObject(ctypes.Structure):
        pass
    _PyObject._fields_ = [
        ('_ob_next', ctypes.POINTER(_PyObject)),
        ('_ob_prev', ctypes.POINTER(_PyObject)),
        ('ob_refcnt', _Py_ssize_t),
        ('ob_type', ctypes.POINTER(_PyObject))
    ]


class _DictProxy(_PyObject):
    _fields_ = [('dict', ctypes.POINTER(_PyObject))]


def reveal_dict(proxy):
    if not isinstance(proxy, MappingProxyType):
        raise TypeError('dictproxy expected')
    dp = _DictProxy.from_address(id(proxy))
    ns = {}
    ctypes.pythonapi.PyDict_SetItem(ctypes.py_object(ns),
                                    ctypes.py_object(None),
                                    dp.dict)
    return ns[None]


def get_class_dict(cls):
    d = getattr(cls, '__dict__', None)
    if d is None:
        raise TypeError('given class does not have a dictionary')
    if isinstance(d, MappingProxyType):
        return reveal_dict(d)
    return d


def test():
    from random import choice
    d = get_class_dict(str)
    d['foo'] = lambda x: ''.join(choice((c.upper, c.lower))() for c in x)
    print("and this is monkey patching str".foo())


if __name__ == '__main__':
    test()

############ and implemented by James on SO ################

import ctypes
from types import MappingProxyType, MethodType


# figure out side of _Py_ssize_t
if hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    _Py_ssize_t = ctypes.c_int64
else:
    _Py_ssize_t = ctypes.c_int


# regular python
class _PyObject(ctypes.Structure):
    pass

_PyObject._fields_ = [
    ('ob_refcnt', _Py_ssize_t),
    ('ob_type', ctypes.POINTER(_PyObject))
]


# python with trace
if object.__basicsize__ != ctypes.sizeof(_PyObject):
    class _PyObject(ctypes.Structure):
        pass
    _PyObject._fields_ = [
        ('_ob_next', ctypes.POINTER(_PyObject)),
        ('_ob_prev', ctypes.POINTER(_PyObject)),
        ('ob_refcnt', _Py_ssize_t),
        ('ob_type', ctypes.POINTER(_PyObject))
    ]


class _DictProxy(_PyObject):
    _fields_ = [('dict', ctypes.POINTER(_PyObject))]


def reveal_dict(proxy):
    if not isinstance(proxy, MappingProxyType):
        raise TypeError('dictproxy expected')
    dp = _DictProxy.from_address(id(proxy))
    ns = {}
    ctypes.pythonapi.PyDict_SetItem(ctypes.py_object(ns),
                                    ctypes.py_object(None),
                                    dp.dict)
    return ns[None]


def get_class_dict(cls):
    d = getattr(cls, '__dict__', None)
    if d is None:
        raise TypeError('given class does not have a dictionary')
    if isinstance(d, MappingProxyType):
        return reveal_dict(d)
    return d


class Listener:
    def __init__(self):
        self._g = None
    def __call__(self, x=None):
        if x is None:
            return self._g
        self._g = x
    def send(self, val):
        if self._g:
            self._g.send(val)


def monkey_patch_list(decorator, mutators=None):
    if not mutators:
        mutators = (
            'append', 'clear', 'extend', 'insert', 'pop', 'remove',
            'reverse', 'sort'
        )
    d_list = get_class_dict(list)
    d_list['_listener'] = Listener()
    for m in mutators:
        d_list[m.capitalize()] = decorator(d_list.get(m))


def before_after(clsm):
	'''decorator for list class methods'''

	def wrapper(self, *args, **kwargs):
		self._listener.send(self)
		out = clsm(self, *args, **kwargs)
		self._listener.send(self)
		return out

	return wrapper


class Watchman:
	def __init__(self):
		self.guarding = []

	def watch(self, lst, fn, name='list'):
		self.guarding.append((lst, name))
		w = self._watcher(fn, name)
		lst._listener(w)

	@staticmethod
	def _watcher(fn, name):
		def gen():
			while True:
				x = yield
				x = str(x)
				y = yield
				y = str(y)
				print(fn(x, y, name=name))

		g = gen()
		next(g)
		return g


def enemies_changed(old, new, name='list'):
    print(f"{name} was {old}, now are {new}")

if __name__ == '__main__':
	# update the list methods with the wrapper
	monkey_patch_list(before_after)

	enemies = ["Moloch", "Twilight Lady", "Big Figure", "Captain Carnage", "Nixon"]
	owl = Watchman()
	owl.watch(enemies, enemies_changed, 'Enemies')

	enemies.Append('Alien')
	# prints:
	# Enemies was ['Moloch', 'Twilight Lady', 'Big Figure', 'Captain Carnage', 'Nixon'],
	# now are ['Moloch', 'Twilight Lady', 'Big Figure', 'Captain Carnage', 'Nixon', 'Alien']
