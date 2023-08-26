
"""Holding general tools for custom syntax and evaluation
of user-defined expressions.

Need to evaluate expressions in any structure, any possibly recursively

"""

from .syntax import SyntaxPasses, ExpTokens, ExpSyntaxError, ExpSyntaxProcessor
from .new import Expression, ExpPolicy, EvaluationError, ExpEvaluator
from .dirty import DirtyExp



