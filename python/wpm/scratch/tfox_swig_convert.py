from maya import OpenMaya as om
from ctypes import c_double
import numpy as np
import time

start = time.time()
# create a dummy array for testing purposes
pa = [float(i) for i in range(1000000)]
util = om.MScriptUtil()
util.createFromList(pa, len(pa))
end = time.time()
print "Allocation and copying took:", end - start


start = time.time()
# cast the scrptUtil object to a swig double pointer
ptr = util.asDoublePtr()
# Cast the swig double pointer to a ctypes array
cta = (c_double * len(pa)).from_address(int(ptr))
# Memory map the ctypes array into numpy
out = np.ctypeslib.as_array(cta)
# ptr, cta, and out are all pointing to the same memory address now
# so changing out will change ptr

# for safety, make a copy of out so I don't corrupt memory
out = np.copy(out)
end = time.time()
print "Getting the numpy interface took:", end - start