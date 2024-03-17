

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

	def topDirStructureIsValid(self):
		"""check if the top dir structure is valid
		"""
		assert self.topPath.exists(), f"top path {self.topPath} does not exist"
		assert self.sourceDir.exists(), f"source dir {self.sourceDir} does not exist"
		assert (self.sourceDir / "__genInit__.py").exists(), f"source dir {self.sourceDir} does not contain __genInit__.py"
		assert (self.sourceDir / "__modifiedInit__.py").exists(), f"source dir {self.sourceDir} does not contain __modifiedInit__.py"


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
					checkFirst=True
	               ):
		"""regenerate full project
		"""
		log(f"regenerating code for project {self.topPath}")
		if checkFirst:
			self.topDirStructureIsValid()
		# first clear out old stuff
		shutil.rmtree(self.refPath, ignore_errors=True)
		self.refPath.mkdir()
		shutil.rmtree(self.genPath, ignore_errors=True)
		self.genPath.mkdir()

		if not self.modifiedPath.exists():
			self.modifiedPath.mkdir()
		log(f"cleared old ref and gen folders")

		# copy over init files
		shutil.copy2(self.sourceDir / "__genInit__.py", self.genPath / "__init__.py",
		            )
		shutil.copy2(self.sourceDir / "__modifiedInit__.py", self.modifiedPath / "__init__.py")

	def populateRefFolder(self, genFn:T.Callable[[CodeGenProject], None]):
		"""populate ref folder with generated files -
		no real need for coupling here
		"""
		genFn(self)

	def mergeGenModified(self):
		"""take files from ref to copy to gen, or merge with modified
		for now just copy everything
		"""
		pass
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


