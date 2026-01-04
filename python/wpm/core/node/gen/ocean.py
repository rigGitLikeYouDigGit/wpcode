

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Texture2d = Catalogue.Texture2d
else:
	from .. import retriever
	Texture2d = retriever.getNodeCls("Texture2d")
	assert Texture2d

# add node doc



# region plug type defs
class ColorModePlug(Plug):
	node : Ocean = None
	pass
class FoamEmissionPlug(Plug):
	node : Ocean = None
	pass
class FoamThresholdPlug(Plug):
	node : Ocean = None
	pass
class NumFrequenciesPlug(Plug):
	node : Ocean = None
	pass
class ObserverSpeedPlug(Plug):
	node : Ocean = None
	pass
class OutFoamPlug(Plug):
	node : Ocean = None
	pass
class ScalePlug(Plug):
	node : Ocean = None
	pass
class TimePlug(Plug):
	node : Ocean = None
	pass
class WaveDirSpreadPlug(Plug):
	node : Ocean = None
	pass
class WaveHeight_FloatValuePlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : Ocean = None
	pass
class WaveHeight_InterpPlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : Ocean = None
	pass
class WaveHeight_PositionPlug(Plug):
	parent : WaveHeightPlug = PlugDescriptor("waveHeight")
	node : Ocean = None
	pass
class WaveHeightPlug(Plug):
	waveHeight_FloatValue_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	whfv_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	waveHeight_Interp_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	whi_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	waveHeight_Position_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	whp_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	node : Ocean = None
	pass
class WaveLengthMaxPlug(Plug):
	node : Ocean = None
	pass
class WaveLengthMinPlug(Plug):
	node : Ocean = None
	pass
class WavePeaking_FloatValuePlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : Ocean = None
	pass
class WavePeaking_InterpPlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : Ocean = None
	pass
class WavePeaking_PositionPlug(Plug):
	parent : WavePeakingPlug = PlugDescriptor("wavePeaking")
	node : Ocean = None
	pass
class WavePeakingPlug(Plug):
	wavePeaking_FloatValue_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wpfv_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wavePeaking_Interp_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wpi_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wavePeaking_Position_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	wpp_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	node : Ocean = None
	pass
class WaveTurbulence_FloatValuePlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : Ocean = None
	pass
class WaveTurbulence_InterpPlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : Ocean = None
	pass
class WaveTurbulence_PositionPlug(Plug):
	parent : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	node : Ocean = None
	pass
class WaveTurbulencePlug(Plug):
	waveTurbulence_FloatValue_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	wtbfv_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	waveTurbulence_Interp_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	wtbi_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	waveTurbulence_Position_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	wtbp_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	node : Ocean = None
	pass
class WindUPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : Ocean = None
	pass
class WindVPlug(Plug):
	parent : WindUVPlug = PlugDescriptor("windUV")
	node : Ocean = None
	pass
class WindUVPlug(Plug):
	windU_ : WindUPlug = PlugDescriptor("windU")
	wiu_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	wiv_ : WindVPlug = PlugDescriptor("windV")
	node : Ocean = None
	pass
# endregion


# define node class
class Ocean(Texture2d):
	colorMode_ : ColorModePlug = PlugDescriptor("colorMode")
	foamEmission_ : FoamEmissionPlug = PlugDescriptor("foamEmission")
	foamThreshold_ : FoamThresholdPlug = PlugDescriptor("foamThreshold")
	numFrequencies_ : NumFrequenciesPlug = PlugDescriptor("numFrequencies")
	observerSpeed_ : ObserverSpeedPlug = PlugDescriptor("observerSpeed")
	outFoam_ : OutFoamPlug = PlugDescriptor("outFoam")
	scale_ : ScalePlug = PlugDescriptor("scale")
	time_ : TimePlug = PlugDescriptor("time")
	waveDirSpread_ : WaveDirSpreadPlug = PlugDescriptor("waveDirSpread")
	waveHeight_FloatValue_ : WaveHeight_FloatValuePlug = PlugDescriptor("waveHeight_FloatValue")
	waveHeight_Interp_ : WaveHeight_InterpPlug = PlugDescriptor("waveHeight_Interp")
	waveHeight_Position_ : WaveHeight_PositionPlug = PlugDescriptor("waveHeight_Position")
	waveHeight_ : WaveHeightPlug = PlugDescriptor("waveHeight")
	waveLengthMax_ : WaveLengthMaxPlug = PlugDescriptor("waveLengthMax")
	waveLengthMin_ : WaveLengthMinPlug = PlugDescriptor("waveLengthMin")
	wavePeaking_FloatValue_ : WavePeaking_FloatValuePlug = PlugDescriptor("wavePeaking_FloatValue")
	wavePeaking_Interp_ : WavePeaking_InterpPlug = PlugDescriptor("wavePeaking_Interp")
	wavePeaking_Position_ : WavePeaking_PositionPlug = PlugDescriptor("wavePeaking_Position")
	wavePeaking_ : WavePeakingPlug = PlugDescriptor("wavePeaking")
	waveTurbulence_FloatValue_ : WaveTurbulence_FloatValuePlug = PlugDescriptor("waveTurbulence_FloatValue")
	waveTurbulence_Interp_ : WaveTurbulence_InterpPlug = PlugDescriptor("waveTurbulence_Interp")
	waveTurbulence_Position_ : WaveTurbulence_PositionPlug = PlugDescriptor("waveTurbulence_Position")
	waveTurbulence_ : WaveTurbulencePlug = PlugDescriptor("waveTurbulence")
	windU_ : WindUPlug = PlugDescriptor("windU")
	windV_ : WindVPlug = PlugDescriptor("windV")
	windUV_ : WindUVPlug = PlugDescriptor("windUV")

	# node attributes

	typeName = "ocean"
	apiTypeInt = 875
	apiTypeStr = "kOcean"
	typeIdInt = 1381257027
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["colorMode", "foamEmission", "foamThreshold", "numFrequencies", "observerSpeed", "outFoam", "scale", "time", "waveDirSpread", "waveHeight_FloatValue", "waveHeight_Interp", "waveHeight_Position", "waveHeight", "waveLengthMax", "waveLengthMin", "wavePeaking_FloatValue", "wavePeaking_Interp", "wavePeaking_Position", "wavePeaking", "waveTurbulence_FloatValue", "waveTurbulence_Interp", "waveTurbulence_Position", "waveTurbulence", "windU", "windV", "windUV"]
	nodeLeafPlugs = ["colorMode", "foamEmission", "foamThreshold", "numFrequencies", "observerSpeed", "outFoam", "scale", "time", "waveDirSpread", "waveHeight", "waveLengthMax", "waveLengthMin", "wavePeaking", "waveTurbulence", "windUV"]
	pass

