


class CompilationError(Exception):
	"""denotes invalid code passed to expression"""
	pass

class EvaluationError(Exception):
	"""denotes error during evaluation of expression"""
	pass

class ExpSyntaxError(Exception):
	"""denotes invalid syntax in expression"""
	pass

