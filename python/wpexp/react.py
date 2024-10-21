
from __future__ import  annotations
import typing as T


#todo: put this somewhere
#todo: unify with the reactive tests, I think it should all be the same thing
from param.reactive import rx
from types import FunctionType, MethodType
def EVAL(subject):
	"""single-shot function to call on anything,
	to retrieve a static value:
	if it's a function, call it
		how do we pass in variables to the function?
		with EvalGlobals{a : b, self : self} as g:
			EVAL(exp) ?
	if it's an rx chain, get its value
	else just return the input

	TODO: just checking for callable isn't enough, since we use call for pathing syntax in trees
	 - consider a default map of types that can be eval'd
	 - mixin with instance method to control further
	"""
	if isinstance(subject, (FunctionType, MethodType)): return subject()
	return subject

def BIND(subject, outputFn:callable):
	pass

from PySide2 import QtCore, QtWidgets, QtGui
from threading import Thread
import time, math

from wplib import log

def sinTime():
	return math.sin(time.time())

class WRX(rx):
	def __str__(self):
		return f"WRX({self.rx.value})"

run = True

val = "INIT VAL"
rxval = WRX(val)
def task():
	while run:
		print("time", sinTime())
		rxval.rx.value = str(sinTime())
		#rxval.rx = str(sinTime())
		print("val", rxval.rx.value)
		time.sleep(0.2)
	pass

# def makeWidget():
# 	line = QtWidgets.QLineEdit()
# 	rxline = WRX(line)
# 	rxline.setText(rxval)
# 	log(rxline)
# 	log(rxline.setText)
# 	log(rxline.text().rx.value)
# 	rxline.setText(val)
# 	line.setText(val)
#
#
# 	return line
#
#
# if __name__ == '__main__':
#
# 	app = QtWidgets.QApplication()
# 	w = makeWidget()
# 	w.show()
#
# 	t = Thread(target=task)
# 	#t.start()
# 	#print("after run")
# 	#time.sleep(5)
# 	#print("after end")
# 	#run = False
# 	# t.join(timeout=1)
#
# 	app.exec_()
#
#


from param.reactive import rx
from PySide2 import QtWidgets # or PySide6, I get the same result

text = "initial_value" # raw str text value
rxtext = rx(text)

# create the Qt application, required to start building widgets
# event loop doesn't run yet
app = QtWidgets.QApplication()


line = QtWidgets.QLineEdit() # create a QLineEdit widget instance
rxline = rx(line) # wrap widget instance in rx - which should also wrap all its methods
# by default QLineEdit has no displayed text

rxline.setText(rxtext) # link rx wrappers together
# observe that the widget still has no text ._.

# if we just call the normal method on the normal instance, obviously it works
# uncomment the line below to see desired result
#line.setText(text)
#display the widget
line.show()

rxtext.rx.value = "new val"
# run the Qt application
app.exec_()








