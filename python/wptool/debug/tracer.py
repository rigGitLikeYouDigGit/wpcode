from __future__ import annotations
import types, typing as T
import pprint
import sys
import functools
from wplib import log


class Tracer:
	"""Traces function calls with indented output."""

	_active_tracer = None  # Class variable to track active tracer

	def __init__(self, whitelist: T.Optional[T.List[str]] = None):
		self.depth = 0
		self._original_trace = None
		self._is_active = False
		self.whitelist = whitelist  # List of package prefixes to trace

	def _should_trace(self, filename: str) -> bool:
		"""Check if the file should be traced based on whitelist."""
		if self.whitelist is None:
			return True

		# Normalize path separators
		filename = filename.replace("\\", ".").replace("/", ".")

		# Check if filename matches any whitelisted package
		for package in self.whitelist:
			if package in filename:
				return True
		return False

	def _trace_calls(self, frame, event, arg):
		if event == "call":
			code = frame.f_code
			func_name = code.co_name
			filename = code.co_filename
			lineno = frame.f_lineno

			# Check if we should trace this file
			if not self._should_trace(filename):
				self.depth += 1
				return self._trace_calls

			# Format filename for display
			display_filename = filename[:filename.rfind("/", )] if "/" in filename else filename

			# Get function arguments
			arg_names = code.co_varnames[:code.co_argcount]
			args = []
			for arg_name in arg_names:
				if arg_name in frame.f_locals:
					arg_value = frame.f_locals[arg_name]
					args.append(f"{arg_name}={repr(arg_value)}")

			args_str = ", ".join(args) if args else ""

			indent = "  " * self.depth
			print(f"{indent}-> {func_name}({args_str}) [{display_filename}:{lineno}]")
			self.depth += 1

		elif event == "return":
			self.depth -= 1

		return self._trace_calls

	def start(self):
		"""Start tracing function calls."""
		if Tracer._active_tracer is not None:
			# Another tracer is already active, don't interfere
			return
		self._original_trace = sys.gettrace()
		sys.settrace(self._trace_calls)
		self._is_active = True
		Tracer._active_tracer = self

	def stop(self):
		"""Stop tracing function calls."""
		if not self._is_active:
			# This tracer was never started (reentrant call)
			return
		sys.settrace(self._original_trace)
		self.depth = 0
		self._is_active = False
		Tracer._active_tracer = None

	def __enter__(self):
		self.start()
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.stop()

	@classmethod
	def trace(cls, func, whitelist: T.Optional[T.List[str]] = None):
		"""Decorator to trace a specific function and all its nested calls."""
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			tracer = cls(whitelist=whitelist)
			tracer.start()
			try:
				result = func(*args, **kwargs)
				return result
			finally:
				tracer.stop()
		return wrapper


