
"""Holding general tools for custom syntax and evaluation
of user-defined expressions.

Need to evaluate expressions in any structure, any possibly recursively

holds a few features -
	- reactive chainable objects to create function pipelines
	- parsing string syntax into expressions and filters
	- matching strings against listing filters

in time it might make more sense to split these up and move them,
still unsure of a lot of it

"""

from .syntax import SyntaxPasses, ExpTokens#, ExpSyntaxError, ExpSyntaxProcessor
from .new import Expression, ExpPolicy#, EvaluationError, ExpEvaluator
from .dirty import DirtyExp

from .react import *

