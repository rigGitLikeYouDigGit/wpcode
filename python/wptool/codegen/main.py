

from __future__ import annotations
import typing as T

import sys, os, json, shutil
from pathlib import Path

from wplib import log

class TemplateDir:
	"""LATER, use tree subclass to allow more complex templates
	"""


class CodeGenProject:
	""" a project for code generation

	specific structure enforced:
	targetTopDir/
		- ref/
		- modified/
		- gen/
		- source/
			- __genInit__.py
			- __modifiedInit__.py

	template file and parametres passed to separate functions
	first param is name of files to create

	only single-file depth for now

	"""

	def __init__(self, targetTopDir:Path):
		"""initialise a project on a target path -
		path need not exist

		"""
		self.topPath = targetTopDir

	@property
	def refPath(self)->Path:
		return self.topPath / "ref"

	@property
	def modifiedPath(self)->Path:
		return self.topPath / "modified"

	@property
	def genPath(self)->Path:
		return self.topPath / "gen"

	@property
	def sourceDir(self)->Path:
		return self.topPath / "source"

	def processTemplate(self, templateFile:Path,
	                    newName:str,
	                    newPath:Path,):
		"""process a template file
		"""
		pass

	def mergeModifiedWithNewGen(self,
	                            oldModifiedPath:Path,
	                            newGenPath:Path):
		"""merge modified with new gen,
		probably delegate to a merger object or something
		"""
		pass

	def regenerate(self,

	               ):
		"""regenerate full project
		"""
		log(f"regenerating code for project {self.topPath}")
		# first clear out old stuff
		shutil.rmtree(self.refPath)
		self.refPath.mkdir()
		shutil.rmtree(self.genPath)
		self.genPath.mkdir()
		log(f"cleared old ref and gen folders")

		# copy over init files
		shutil.copy(self.sourceDir / "__genInit__.py", self.genPath / "__init__.py")
		shutil.copy(self.sourceDir / "__modifiedInit__.py", self.modifiedPath / "__init__.py")
		#
		# # first populate ref folder - all generated files
		# for name in fileNames:
		# 	newPath = self.refPath / name
		# 	# copy file
		# 	shutil.copy(templateFile, newPath)
		# 	# process file, generate code
		# 	self.processTemplate(templateFile, name, newPath)
		#
		# # compare ref to modified, copy to gen if not found
		# for refFile in self.refPath.iterdir():
		# 	modifiedFile = self.modifiedPath / refFile.name
		# 	if not modifiedFile.exists():
		# 		shutil.copy(refFile, self.genPath / refFile.name)
		#
		# # merge any files in modified with their ref counterparts
		# for modifiedFile in self.modifiedPath.iterdir():
		# 	refFile = self.refPath / modifiedFile.name
		# 	if refFile.exists():
		# 		self.mergeModifiedWithNewGen(modifiedFile, refFile)


