
from __future__ import annotations
import typing as T

from wplib import coerce

from wpm import WN, createWN, cmds, om, WPlug, PLUG_VAL


""" module for rigs relating to tension - pulleys, ropes,
winches, hanging cables

testing new convention for functions - 
 first argument is string name
 
ignore fancy object interfaces for complex rig structures for now

AXIS of any rotational transform is X
"""



@coerce
def homotheticCentres(
		name:str,
		posA:PlugTerm,
		posB:PlugTerm,
		radiusA:PlugTerm,
		radiusB:PlugTerm,
):
	"""return vectors for inner and outer homothetic centres
	between 2 circles of different radii
	"""

	outerSum = radiusA + radiusB
	innerSum = radiusA - radiusB
	centreDistance = posA.distanceTo(posB)
	pass

def _createPulleyOutgoingAngle(
		name:str,
		centreDistance:PLUG_VAL,
		pulleyRadius:PLUG_VAL,
		destRadius:PLUG_VAL,
):
	"""create single offset angle for outgoing rope,
	centreDistance : e
	radius : r1, r2

	"""

	angle = arcsin((destRadius - pulleyRadius) / centreDistance)



def createPulleyBeltCurve(
		name:str,
		pulleyTfs:list[WN],
		pulleyRadii:list[PLUG_VAL])->dict[str, WN]:
	"""Passing transforms and radii separately
	feels clunky but it's simple"""







