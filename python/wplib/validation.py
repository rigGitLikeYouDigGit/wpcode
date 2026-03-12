
from __future__ import annotations
import typing as T

from dataclasses import dataclass

from wplib.errorreport import ErrorReport

class ValidationError(Exception):
	pass



"""
state:
valid
errored - no fix
auto-corrected
invalid but fixable on confirmation

For now, rules have to be distinct classes - if needed, a general
subclass could just accept functions from user

error report class might be redundant here, only using it to 
catch exceptions raised in rule code

"""



class ValidationRule:
	"""model on vfx qc checks - each rule checks a specific aspect of the input, and can optionally suggest a fix if the input fails the check. Rules are then grouped into RuleSets which can be applied to inputs and return a list of ValidationResults for each rule check.

	test using ints for status? that way we can check that a set of rules
	doesn't go above a certain severity level, e.g. auto-correction is fine but anything that needs user confirmation is not."""

	VALID = 0
	AUTO_CORRECTED = 1
	ERRORED_BUT_FIXABLE = 2
	ERRORED = 3

	ValidationResult = ValidationResult

	def __init__(self, name:str="", *args, **kwargs):
		self.name = name or self.__class__.__name__
		self.args = args
		self.kwargs = kwargs

	def checkInput(self, data, *args, **kwargs)->int:
		"""override - check input against rule, return severity level
		of error
		"""
		raise NotImplementedError

	def isCorrectable(self, data, *args, **kwargs)->bool:
		"""override - return True if input fails rule check but can be auto-corrected"""
		return False

	def isFixable(self, data, *args, **kwargs)->bool:
		"""override - return True if input fails rule check but can be fixed by user confirmation"""
		return False

	def getSuggestedValue(self, data, *args, **kwargs)->T.Any:
		"""override - return a suggested value to replace
		invalid input, may be shown adaptively as user types"""
		raise NotImplementedError

	def getMessage(self, data,  *args, **kwargs)->str:
		"""override - return a message to show to the user if input fails
		this rule check, and why"""
		return f"Input {data} failed rule {self.name}"

	@classmethod
	def checkRules(cls, rules:list[ValidationRule], data, *args,
	               autoFix:bool=None,
	               resultMap:dict[str, ValidationResult]=None,
	               **kwargs
	               )->bool:
		"""check input against all rules, return True if all pass.
		If any fail (or an error is raised), return False.

		autoFix tristate: by default, any manually-fixable error will be
		returned as an error.
		    If autoFix is True, these will be manually fixed and check will
		    succeed
		as auto-corrected with the suggested fix in the resultMap.
			If autoFix is False, even auto-correctable error will be returned as an error.
		autofix could also be made a set of rule names
		Calling code can then inspect resultMap for rich results
		"""
		results = resultMap if resultMap is not None else {}
		for rule in rules:
			try:
				status = rule.checkInput(data, *args, **kwargs)
				if status == cls.VALID:
					results[ rule.name] = ValidationResult(
						status)
					continue
				if rule.isCorrectable(data, *args, **kwargs):
					if autoFix is False:
						results[rule.name] = ValidationResult(
							cls.ERRORED,
						)
						continue
					newData = rule.getSuggestedValue(data,*args, **kwargs)
					results[rule.name] = ValidationResult(
						cls.AUTO_CORRECTED,
						suggestion=newData,
						message=rule.getMessage(data, *args, **kwargs)
					)
					data = newData
					continue
				if rule.isFixable(data):
					if autoFix is True:
						newData = rule.getSuggestedValue(data, *args, **kwargs)
						results[rule.name] = ValidationResult(
							cls.AUTO_CORRECTED,
							suggestion=newData,
							message=rule.getMessage(data, *args, **kwargs)
						)
						data = newData
						continue

					results[rule.name] = ValidationResult(
						cls.ERRORED_BUT_FIXABLE,
						suggestion=rule.getSuggestedValue(data, *args, **kwargs),
						message=rule.getMessage(data, *args, **kwargs)
					)
					continue
				results[rule.name] = ValidationResult(
					cls.ERRORED,
					message=rule.getMessage(data, *args, **kwargs)
				)

			except Exception as e:
				results[rule.name] = ValidationResult(
					cls.ERRORED,
					errorReport=ErrorReport(
						e,
						f"Error validating input {data} \n\t with rule "
						   f"{rule.name}, fix code as priority "
						),
					suggestion=f"Error validating input {data} \n\t with rule "
						   f"{rule.name}, fix code as priority "
				)
				continue

		return max(result.status for result in results.values()) in (cls.VALID, cls.AUTO_CORRECTED)



@dataclass
class ValidationResult:
	"""result of a validation check -
	return success or failure, descriptive error,
	optionally a suggested replacement value"""
	status : int = ValidationRule.VALID
	errorReport : ErrorReport = None
	suggestion : T.Any = None
	message : str = ""


class RuleSet:

	def __init__(self, rules:list[ValidationRule]):
		self.rules = rules
		self.results = [] # feels weird to give this object state

	def checkInput(self, data)->str:
		"""check input against all rules, return True if all pass.
		If any fail (or an error is raised), return False.
		Calling code can then inspect self.results for more info"""
		self.results = []
		for rule in self.rules:
			try:
				result = rule.checkInput(data)
				if result == True or result is None:
					self.results.append(ValidationResult())
			except ValidationError as e: # input failed rule check
				suggestion = rule.getSuggestedValue(data)
				self.results.append(
					ValidationResult(
						success=False,
						errorReport=ErrorReport(
							e, f"Validation error for input {data} \n\t with rule {rule.name} "
							),
						suggestion=suggestion
					                 )
				)
			except Exception as e: # something gone wrong in actual rule code
				self.results.append(
					ValidationResult(success=False,
					errorReport=ErrorReport(
						e, f"Error validating input {data} \n\t with rule {rule.name} "
						)
					                 )
				)
		#print("results", self.results)
		successStatuses = [result.success for result in self.results]
		return all(successStatuses)

