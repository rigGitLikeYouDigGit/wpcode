from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

"""
define simple classes for all the event message constants in maya 

TODO: - it would be nice to have these hinted on the actual
api class itself, maybe it's worth adding it to the stubs
we get from maya directly

watch dark city
"""

"""
from wpm.lib.generate import generateClassStub
#print(str(generateClassStub(msg)))
for i in [msg] + msg.__subclasses__():
    print(str(generateClassStub(i)))

"""

class MMessage(object):
	# currentCallbackId : staticmethod = <staticmethod object at 0x000002F149F25E80>
	# nodeCallbacks : staticmethod = <staticmethod object at 0x000002F149F25E50>
	# removeCallback : staticmethod = <staticmethod object at 0x000002F149F25E20>
	# removeCallbacks : staticmethod = <staticmethod object at 0x000002F149F25DF0>
	__doc__ : str = "Base class for message callbacks."
	kDefaultAction : int = 0
	kDoNotDoAction : int = 1
	kDoAction : int = 2
	pass
class MCameraMessage(MMessage):
	# addBeginManipulationCallback : staticmethod = <staticmethod object at 0x000002F149F25D90>
	# addEndManipulationCallback : staticmethod = <staticmethod object at 0x000002F149F25D60>
	__doc__ : str = "Class used to register callbacks for Camera Manipulation Begin and End related messages."
	pass
class MCommandMessage(MMessage):
	# addCommandCallback : staticmethod = <staticmethod object at 0x000002F149F25CD0>
	# addProcCallback : staticmethod = <staticmethod object at 0x000002F149F25CA0>
	# addCommandOutputCallback : staticmethod = <staticmethod object at 0x000002F149F25C70>
	# addCommandOutputFilterCallback : staticmethod = <staticmethod object at 0x000002F149F25C10>
	__doc__ : str = """Class used to register callbacks for command related messages.

	The class also provides the following MessageType constants which
	describe the different types of output messages:
	  kHistory		#Command history
	  kDisplay		#String to display unmodified
	  kInfo		#General information
	  kWarning		#Warning message
	  kError		#Error message
	  kResult		#Result from a command execution in the command window
	  kStackTrace	#Stack trace"""
	kHistory : int = 0
	kDisplay : int = 1
	kInfo : int = 2
	kWarning : int = 3
	kError : int = 4
	kResult : int = 5
	kStackTrace : int = 6
	kMELProc : int = 0
	kMELCommand : int = 1
	pass
class MConditionMessage(MMessage):
	# addConditionCallback : staticmethod = <staticmethod object at 0x000002F149F25880>
	# getConditionNames : staticmethod = <staticmethod object at 0x000002F149F258E0>
	# getConditionState : staticmethod = <staticmethod object at 0x000002F149FB5040>
	__doc__ : str = "Class used to register callbacks for condition related messages."
	pass
class MContainerMessage(MMessage):
	# addPublishAttrCallback : staticmethod = <staticmethod object at 0x000002F149FB50A0>
	# addBoundAttrCallback : staticmethod = <staticmethod object at 0x000002F149FB50D0>
	__doc__ : str = "Class used to register callbacks for container related messages."
	pass
class MDagMessage(MMessage):
	# addParentAddedCallback : staticmethod = <staticmethod object at 0x000002F149FB5130>
	# addParentAddedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5160>
	# addParentRemovedCallback : staticmethod = <staticmethod object at 0x000002F149FB5190>
	# addParentRemovedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB51C0>
	# addChildAddedCallback : staticmethod = <staticmethod object at 0x000002F149FB51F0>
	# addChildAddedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5220>
	# addChildRemovedCallback : staticmethod = <staticmethod object at 0x000002F149FB5250>
	# addChildRemovedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5280>
	# addChildReorderedCallback : staticmethod = <staticmethod object at 0x000002F149FB52B0>
	# addChildReorderedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB52E0>
	# addDagCallback : staticmethod = <staticmethod object at 0x000002F149FB5310>
	# addDagDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5340>
	# addAllDagChangesCallback : staticmethod = <staticmethod object at 0x000002F149FB5370>
	# addAllDagChangesDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB53A0>
	# addInstanceAddedCallback : staticmethod = <staticmethod object at 0x000002F149FB53D0>
	# addInstanceAddedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5400>
	# addInstanceRemovedCallback : staticmethod = <staticmethod object at 0x000002F149FB5430>
	# addInstanceRemovedDagPathCallback : staticmethod = <staticmethod object at 0x000002F149FB5460>
	# addWorldMatrixModifiedCallback : staticmethod = <staticmethod object at 0x000002F149FB5490>
	# addMatrixModifiedCallback : staticmethod = <staticmethod object at 0x000002F149FB54C0>
	__doc__ : str = """Class used to register callbacks for Dag related messages.

	The class also provides the following DagMessage constants which describe the different types of DAG operations:
	  kParentAdded
	  kParentRemoved
	  kChildAdded
	  kChildRemoved
	  kChildReordered
	  kInstanceAdded
	  kInstanceRemoved
	  kInvalidMsg"""

	kInvalidMsg : int = -1
	kParentAdded : int = 0
	kParentRemoved : int = 1
	kChildAdded : int = 2
	kChildRemoved : int = 3
	kChildReordered : int = 4
	kInstanceAdded : int = 5
	kInstanceRemoved : int = 6
	kLast : int = 7
	kScaleX : int = 1
	kScaleY : int = 2
	kScaleZ : int = 4
	kShearXY : int = 8
	kShearXZ : int = 16
	kShearYZ : int = 32
	kRotateX : int = 64
	kRotateY : int = 128
	kRotateZ : int = 256
	kTranslateX : int = 512
	kTranslateY : int = 1024
	kTranslateZ : int = 2048
	kScalePivotX : int = 4096
	kScalePivotY : int = 8192
	kScalePivotZ : int = 16384
	kRotatePivotX : int = 32768
	kRotatePivotY : int = 65536
	kRotatePivotZ : int = 131072
	kScaleTransX : int = 262144
	kScaleTransY : int = 524288
	kScaleTransZ : int = 1048576
	kRotateTransX : int = 2097152
	kRotateTransY : int = 4194304
	kRotateTransZ : int = 8388608
	kRotateOrientX : int = 16777216
	kRotateOrientY : int = 33554432
	kRotateOrientZ : int = 67108864
	kRotateOrder : int = 134217728
	kAll : int = 268435455
	kScale : int = 7
	kShear : int = 56
	kRotation : int = 448
	kTranslation : int = 3584
	kScalePivot : int = 28672
	kRotatePivot : int = 229376
	kScalePivotTrans : int = 1835008
	kRotatePivotTrans : int = 14680064
	kRotateOrient : int = 117440512
	pass
class MDGMessage(MMessage):
	# addTimeChangeCallback : staticmethod = <staticmethod object at 0x000002F149FB5520>
	# addDelayedTimeChangeCallback : staticmethod = <staticmethod object at 0x000002F149FB5550>
	# addDelayedTimeChangeRunupCallback : staticmethod = <staticmethod object at 0x000002F149FB5580>
	# addForceUpdateCallback : staticmethod = <staticmethod object at 0x000002F149FB55B0>
	# addNodeAddedCallback : staticmethod = <staticmethod object at 0x000002F149FB55E0>
	# addNodeRemovedCallback : staticmethod = <staticmethod object at 0x000002F149FB5610>
	# addPreConnectionCallback : staticmethod = <staticmethod object at 0x000002F149FB5640>
	# addConnectionCallback : staticmethod = <staticmethod object at 0x000002F149FB5670>
	# addNodeChangeUuidCheckCallback : staticmethod = <staticmethod object at 0x000002F149FB56A0>
	__doc__ : str = "Class used to register callbacks for Dependency Graph related messages."
	pass
class MEventMessage(MMessage):
	# addEventCallback : staticmethod = <staticmethod object at 0x000002F149FB5700>
	# getEventNames : staticmethod = <staticmethod object at 0x000002F149FB5730>
	__doc__ : str = """Class used to register callbacks for event related messages.

	The first parameter passed to the add callback method is the name
	of the event that will trigger the callback.  The list of
	available event names can be retrieved by calling the
	getEventNames method or by using the -listEvents flag on the
	scriptJob command.
	The addEventCallback method returns an id which is used to remove the
	callback.

	To remove a callback use OpenMaya.MMessage.removeCallback.

	All callbacks that are registered by a plug-in must be removed by
	that plug-in when it is unloaded.  Failure to do so will result in
	a fatal error.

	Idle event callbacks should be removed immediately after running.
	Otherwise they will continue to use up CPU resources. They will also
	prevent idleVeryLow event callbacks from running - which are required
	for Maya to function properly."""

	dbTraceChanged = 'dbTraceChanged'
	resourceLimitStateChange = 'resourceLimitStateChange'
	linearUnitChanged = 'linearUnitChanged'
	timeUnitChanged = 'timeUnitChanged'
	angularUnitChanged = 'angularUnitChanged'
	Undo = 'Undo'
	undoSupressed = 'undoSupressed'
	Redo = 'Redo'
	customEvaluatorChanged = 'customEvaluatorChanged'
	serialExecutorFallback = 'serialExecutorFallback'
	timeChanged = 'timeChanged'
	currentContainerChange = 'currentContainerChange'
	quitApplication = 'quitApplication'
	idleHigh = 'idleHigh'
	idle = 'idle'
	idleVeryLow = 'idleVeryLow'
	RecentCommandChanged = 'RecentCommandChanged'
	ToolChanged = 'ToolChanged'
	PostToolChanged = 'PostToolChanged'
	ToolDirtyChanged = 'ToolDirtyChanged'
	ToolSettingsChanged = 'ToolSettingsChanged'
	tabletModeChanged = 'tabletModeChanged'
	DisplayRGBColorChanged = 'DisplayRGBColorChanged'
	animLayerRebuild = 'animLayerRebuild'
	animLayerRefresh = 'animLayerRefresh'
	animLayerAnimationChanged = 'animLayerAnimationChanged'
	animLayerLockChanged = 'animLayerLockChanged'
	animLayerBaseLockChanged = 'animLayerBaseLockChanged'
	animLayerGhostChanged = 'animLayerGhostChanged'
	cteEventKeyingTargetForClipChanged = 'cteEventKeyingTargetForClipChanged'
	cteEventKeyingTargetForLayerChanged = 'cteEventKeyingTargetForLayerChanged'
	cteEventKeyingTargetForInvalidChanged = 'cteEventKeyingTargetForInvalidChanged'
	teClipAdded = 'teClipAdded'
	teClipModified = 'teClipModified'
	teClipRemoved = 'teClipRemoved'
	teCompositionAdded = 'teCompositionAdded'
	teCompositionRemoved = 'teCompositionRemoved'
	teCompositionActiveChanged = 'teCompositionActiveChanged'
	teCompositionNameChanged = 'teCompositionNameChanged'
	teMuteChanged = 'teMuteChanged'
	cameraChange = 'cameraChange'
	cameraDisplayAttributesChange = 'cameraDisplayAttributesChange'
	GhostListChanged = 'GhostListChanged'
	SelectionChanged = 'SelectionChanged'
	UFESelectionChanged = 'UFESelectionChanged'
	PreSelectionChangedTriggered = 'PreSelectionChangedTriggered'
	LiveListChanged = 'LiveListChanged'
	ActiveViewChanged = 'ActiveViewChanged'
	SelectModeChanged = 'SelectModeChanged'
	SelectTypeChanged = 'SelectTypeChanged'
	SelectPreferenceChanged = 'SelectPreferenceChanged'
	DisplayPreferenceChanged = 'DisplayPreferenceChanged'
	DagObjectCreated = 'DagObjectCreated'
	transformLockChange = 'transformLockChange'
	renderLayerManagerChange = 'renderLayerManagerChange'
	renderLayerChange = 'renderLayerChange'
	displayLayerManagerChange = 'displayLayerManagerChange'
	displayLayerAdded = 'displayLayerAdded'
	displayLayerDeleted = 'displayLayerDeleted'
	displayLayerVisibilityChanged = 'displayLayerVisibilityChanged'
	displayLayerChange = 'displayLayerChange'
	renderPassChange = 'renderPassChange'
	renderPassSetChange = 'renderPassSetChange'
	renderPassSetMembershipChange = 'renderPassSetMembershipChange'
	passContributionMapChange = 'passContributionMapChange'
	DeferredMaterialLoadingBegin = 'DeferredMaterialLoadingBegin'
	DeferredMaterialLoadingDone = 'DeferredMaterialLoadingDone'
	DisplayColorChanged = 'DisplayColorChanged'
	lightLinkingChanged = 'lightLinkingChanged'
	lightLinkingChangedNonSG = 'lightLinkingChangedNonSG'
	UvTileProxyDirtyChangeTrigger = 'UvTileProxyDirtyChangeTrigger'
	preferredRendererChanged = 'preferredRendererChanged'
	polyTopoSymmetryValidChanged = 'polyTopoSymmetryValidChanged'
	SceneSegmentChanged = 'SceneSegmentChanged'
	PostSceneSegmentChanged = 'PostSceneSegmentChanged'
	SequencerActiveShotChanged = 'SequencerActiveShotChanged'
	SoundNodeAdded = 'SoundNodeAdded'
	SoundNodeRemoved = 'SoundNodeRemoved'
	ColorIndexChanged = 'ColorIndexChanged'
	deleteAll = 'deleteAll'
	NameChanged = 'NameChanged'
	symmetricModellingOptionsChanged = 'symmetricModellingOptionsChanged'
	softSelectOptionsChanged = 'softSelectOptionsChanged'
	SetModified = 'SetModified'
	xformConstraintOptionsChanged = 'xformConstraintOptionsChanged'
	undoXformCmd = 'undoXformCmd'
	redoXformCmd = 'redoXformCmd'
	linearToleranceChanged = 'linearToleranceChanged'
	angularToleranceChanged = 'angularToleranceChanged'
	nurbsToPolygonsPrefsChanged = 'nurbsToPolygonsPrefsChanged'
	nurbsCurveRebuildPrefsChanged = 'nurbsCurveRebuildPrefsChanged'
	constructionHistoryChanged = 'constructionHistoryChanged'
	threadCountChanged = 'threadCountChanged'
	SceneSaved = 'SceneSaved'
	NewSceneOpened = 'NewSceneOpened'
	SceneOpened = 'SceneOpened'
	SceneImported = 'SceneImported'
	PreFileNewOrOpened = 'PreFileNewOrOpened'
	PreFileNew = 'PreFileNew'
	PreFileOpened = 'PreFileOpened'
	PostSceneRead = 'PostSceneRead'
	renderSetupAutoSave = 'renderSetupAutoSave'
	workspaceChanged = 'workspaceChanged'
	metadataVisualStatusChanged = 'metadataVisualStatusChanged'
	freezeOptionsChanged = 'freezeOptionsChanged'
	nurbsToSubdivPrefsChanged = 'nurbsToSubdivPrefsChanged'
	selectionConstraintsChanged = 'selectionConstraintsChanged'
	PolyUVSetChanged = 'PolyUVSetChanged'
	PolyUVSetDeleted = 'PolyUVSetDeleted'
	startColorPerVertexTool = 'startColorPerVertexTool'
	stopColorPerVertexTool = 'stopColorPerVertexTool'
	start3dPaintTool = 'start3dPaintTool'
	stop3dPaintTool = 'stop3dPaintTool'
	DragRelease = 'DragRelease'
	ModelPanelSetFocus = 'ModelPanelSetFocus'
	modelEditorChanged = 'modelEditorChanged'
	gridDisplayChanged = 'gridDisplayChanged'
	interactionStyleChanged = 'interactionStyleChanged'
	axisAtOriginChanged = 'axisAtOriginChanged'
	CurveRGBColorChanged = 'CurveRGBColorChanged'
	SelectPriorityChanged = 'SelectPriorityChanged'
	snapModeChanged = 'snapModeChanged'
	MenuModeChanged = 'MenuModeChanged'
	texWindowEditorImageBaseColorChanged = 'texWindowEditorImageBaseColorChanged'
	texWindowEditorCheckerDensityChanged = 'texWindowEditorCheckerDensityChanged'
	texWindowEditorCheckerDisplayChanged = 'texWindowEditorCheckerDisplayChanged'
	texWindowEditorDisplaySolidMapChanged = 'texWindowEditorDisplaySolidMapChanged'
	texWindowEditorShowup = 'texWindowEditorShowup'
	texWindowEditorClose = 'texWindowEditorClose'
	activeHandleChanged = 'activeHandleChanged'
	ChannelBoxLabelSelected = 'ChannelBoxLabelSelected'
	colorMgtOCIORulesEnabledChanged = 'colorMgtOCIORulesEnabledChanged'
	colorMgtUserPrefsChanged = 'colorMgtUserPrefsChanged'
	RenderSetupSelectionChanged = 'RenderSetupSelectionChanged'
	colorMgtEnabledChanged = 'colorMgtEnabledChanged'
	colorMgtConfigFileEnableChanged = 'colorMgtConfigFileEnableChanged'
	colorMgtConfigFilePathChanged = 'colorMgtConfigFilePathChanged'
	colorMgtConfigChanged = 'colorMgtConfigChanged'
	colorMgtWorkingSpaceChanged = 'colorMgtWorkingSpaceChanged'
	colorMgtPrefsViewTransformChanged = 'colorMgtPrefsViewTransformChanged'
	colorMgtPrefsReloaded = 'colorMgtPrefsReloaded'
	colorMgtOutputChanged = 'colorMgtOutputChanged'
	colorMgtPlayblastOutputChanged = 'colorMgtPlayblastOutputChanged'
	colorMgtRefreshed = 'colorMgtRefreshed'
	selectionPipelineChanged = 'selectionPipelineChanged'
	glFrameTrigger = 'glFrameTrigger'
	activeTexHandleChanged = 'activeTexHandleChanged'
	EditModeChanged = 'EditModeChanged'
	graphEditorChanged = 'graphEditorChanged'
	graphEditorParamCurveSelected = 'graphEditorParamCurveSelected'
	graphEditorOutlinerHighlightChanged = 'graphEditorOutlinerHighlightChanged'
	graphEditorOutlinerListChanged = 'graphEditorOutlinerListChanged'
	currentSoundNodeChanged = 'currentSoundNodeChanged'
	playbackRangeAboutToChange = 'playbackRangeAboutToChange'
	playbackSpeedChanged = 'playbackSpeedChanged'
	playbackModeChanged = 'playbackModeChanged'
	playbackRangeSliderChanged = 'playbackRangeSliderChanged'
	playbackByChanged = 'playbackByChanged'
	playbackRangeChanged = 'playbackRangeChanged'
	profilerSelectionChanged = 'profilerSelectionChanged'
	RenderViewCameraChanged = 'RenderViewCameraChanged'
	texScaleContextOptionsChanged = 'texScaleContextOptionsChanged'
	texRotateContextOptionsChanged = 'texRotateContextOptionsChanged'
	texMoveContextOptionsChanged = 'texMoveContextOptionsChanged'
	polyCutUVSteadyStrokeChanged = 'polyCutUVSteadyStrokeChanged'
	polyCutUVEventTexEditorCheckerDisplayChanged = 'polyCutUVEventTexEditorCheckerDisplayChanged'
	polyCutUVShowTextureBordersChanged = 'polyCutUVShowTextureBordersChanged'
	polyCutUVShowUVShellColoringChanged = 'polyCutUVShowUVShellColoringChanged'
	shapeEditorTreeviewSelectionChanged = 'shapeEditorTreeviewSelectionChanged'
	poseEditorTreeviewSelectionChanged = 'poseEditorTreeviewSelectionChanged'
	sculptMeshCacheBlendShapeListChanged = 'sculptMeshCacheBlendShapeListChanged'
	sculptMeshCacheCloneSourceChanged = 'sculptMeshCacheCloneSourceChanged'
	RebuildUIValues = 'RebuildUIValues'
	cacheDestroyed = 'cacheDestroyed'
	cachingPreferencesChanged = 'cachingPreferencesChanged'
	cachingSafeModeChanged = 'cachingSafeModeChanged'
	cachingEvaluationModeChanged = 'cachingEvaluationModeChanged'
	teTrackAdded = 'teTrackAdded'
	teTrackRemoved = 'teTrackRemoved'
	teTrackNameChanged = 'teTrackNameChanged'
	teTrackModified = 'teTrackModified'
	cteEventClipEditModeChanged = 'cteEventClipEditModeChanged'
	teEditorPrefsChanged = 'teEditorPrefsChanged'

	pass
class MLockMessage(MMessage):
	# setNodeLockDAGQueryCallback : staticmethod = <staticmethod object at 0x000002F149FB5790>
	# setNodeLockQueryCallback : staticmethod = <staticmethod object at 0x000002F149FB57C0>
	# setPlugLockQueryCallback : staticmethod = <staticmethod object at 0x000002F149FB57F0>
	__doc__ : str =" Class used to register callbacks for model related messages."
	kInvalidPlug : int = 0
	kPlugLockAttr : int = 1
	kPlugUnlockAttr : int = 2
	kPlugAttrValChange : int = 3
	kPlugRemoveAttr : int = 4
	kPlugRenameAttr : int = 5
	kPlugConnect : int = 6
	kPlugDisconnect : int = 7
	kLastPlug : int = 8
	kInvalidDAG : int = 0
	kGroup : int = 1
	kUnGroup : int = 2
	kReparent : int = 3
	kChildReorder : int = 4
	kCreateNodeInstance : int = 5
	kCreateChildInstance : int = 6
	kCreateParentInstance : int = 7
	kLastDAG : int = 8
	kInvalid : int = 0
	kRename : int = 1
	kDelete : int = 2
	kLockNode : int = 3
	kUnlockNode : int = 4
	kAddAttr : int = 5
	kRemoveAttr : int = 6
	kRenameAttr : int = 7
	kUnlockAttr : int = 8
	kLockAttr : int = 9
	kLast : int = 10
	pass
class MModelMessage(MMessage):
	# addAfterDuplicateCallback : staticmethod = <staticmethod object at 0x000002F149FB5850>
	# addBeforeDuplicateCallback : staticmethod = <staticmethod object at 0x000002F149FB5880>
	# addCallback : staticmethod = <staticmethod object at 0x000002F149FB58B0>
	# addNodeAddedToModelCallback : staticmethod = <staticmethod object at 0x000002F149FB58E0>
	# addNodeRemovedFromModelCallback : staticmethod = <staticmethod object at 0x000002F149FB5910>
	__doc__ : str = """Class used to register callbacks for model related messages.The class also provides the following Message constants which
	describe the different types supported by the addCallback method:
	  kActiveListModified		#active selection changes
"""
	kActiveListModified : int = 0
	pass
class MNodeMessage(MMessage):
	# addAttributeAddedOrRemovedCallback : staticmethod = <staticmethod object at 0x000002F149FB5970>
	# addAttributeChangedCallback : staticmethod = <staticmethod object at 0x000002F149FB59A0>
	# addKeyableChangeOverride : staticmethod = <staticmethod object at 0x000002F149FB59D0>
	# addNameChangedCallback : staticmethod = <staticmethod object at 0x000002F149FB5A00>
	# addNodeAboutToDeleteCallback : staticmethod = <staticmethod object at 0x000002F149FB5A30>
	# addNodeDestroyedCallback : staticmethod = <staticmethod object at 0x000002F149FB5A60>
	# addNodeDirtyCallback : staticmethod = <staticmethod object at 0x000002F149FB5A90>
	# addNodeDirtyPlugCallback : staticmethod = <staticmethod object at 0x000002F149FB5AC0>
	# addNodePreRemovalCallback : staticmethod = <staticmethod object at 0x000002F149FB5AF0>
	# addUuidChangedCallback : staticmethod = <staticmethod object at 0x000002F149FB5B20>
	__doc__ : str = """Class used to register callbacks for dependency node messages of specific dependency nodes.

	The class also provides the following AttributeMessage constants which describe
	the type of attribute changed/addedOrRemoved messages that has occurred:
	  kConnectionMade		#a connection has been made to an attribute of this node
	  kConnectionBroken	#a connection has been broken for an attribute of this node
	  kAttributeEval		#an attribute of this node has been evaluated
	  kAttributeSet		#an attribute value of this node has been set
	  kAttributeLocked		#an attribute of this node has been locked
	  kAttributeUnlocked	#an attribute of this node has been unlocked
	  kAttributeAdded		#an attribute has been added to this node
	  kAttributeRemoved	#an attribute has been removed from this node
	  kAttributeRenamed	#an attribute of this node has been renamed
	  kAttributeKeyable	#an attribute of this node has been marked keyable
	  kAttributeUnkeyable	#an attribute of this node has been marked unkeyable
	  kIncomingDirection	#the connection was coming into the node
	  kAttributeArrayAdded	#an array attribute has been added to this node
	  kAttributeArrayRemoved	#an array attribute has been removed from this node
	  kOtherPlugSet		#the otherPlug data has been set


	The class also provides the following KeyableChangeMsg constants which
	allows user to prevent attributes from becoming (un)keyable:
	  kKeyChangeInvalid
	  kMakeKeyable
	  kMakeUnkeyable
	  kKeyChangeLast
"""
	kConnectionMade : int = 1
	kConnectionBroken : int = 2
	kAttributeEval : int = 4
	kAttributeSet : int = 8
	kAttributeLocked : int = 16
	kAttributeUnlocked : int = 32
	kAttributeAdded : int = 64
	kAttributeRemoved : int = 128
	kAttributeRenamed : int = 256
	kAttributeKeyable : int = 512
	kAttributeUnkeyable : int = 1024
	kIncomingDirection : int = 2048
	kAttributeArrayAdded : int = 4096
	kAttributeArrayRemoved : int = 8192
	kOtherPlugSet : int = 16384
	kLast : int = 32768
	kKeyChangeInvalid : int = 0
	kMakeKeyable : int = 1
	kMakeUnkeyable : int = 2
	kKeyChangeLast : int = 3
	pass
class MObjectSetMessage(MMessage):
	# addSetMembersModifiedCallback : staticmethod = <staticmethod object at 0x000002F149FB5B80>
	__doc__ : str = "Class used to register callbacks for set modified related messages."
	pass
class MPolyMessage(MMessage):
	# addPolyComponentIdChangedCallback : staticmethod = <staticmethod object at 0x000002F149FB5BE0>
	# addPolyTopologyChangedCallback : staticmethod = <staticmethod object at 0x000002F149FB5C10>
	__doc__ : str = """Class used to register callbacks for poly related messages."""
	pass
class MSceneMessage(MMessage):
	# addCallback : staticmethod = <staticmethod object at 0x000002F149FB5C70>
	# addCheckCallback : staticmethod = <staticmethod object at 0x000002F149FB5CA0>
	# addCheckFileCallback : staticmethod = <staticmethod object at 0x000002F149FB5CD0>
	# addCheckReferenceCallback : staticmethod = <staticmethod object at 0x000002F149FB5D00>
	# addConnectionFailedCallback : staticmethod = <staticmethod object at 0x000002F149FB5D30>
	# addReferenceCallback : staticmethod = <staticmethod object at 0x000002F149FB5D60>
	# addStringArrayCallback : staticmethod = <staticmethod object at 0x000002F149FB5D90>
	__doc__ : str = "Class used to register callbacks for scene related messages."
	kSceneUpdate : int = 0
	kBeforeNew : int = 1
	kAfterNew : int = 2
	kBeforeImport : int = 3
	kAfterImport : int = 4
	kBeforeOpen : int = 5
	kAfterOpen : int = 6
	kBeforeFileRead : int = 7
	kAfterFileRead : int = 8
	kAfterSceneReadAndRecordEdits : int = 9
	kBeforeExport : int = 10
	kAfterExport : int = 11
	kBeforeSave : int = 12
	kAfterSave : int = 13
	kBeforeReference : int = 14
	kAfterReference : int = 15
	kBeforeRemoveReference : int = 16
	kAfterRemoveReference : int = 17
	kBeforeImportReference : int = 18
	kAfterImportReference : int = 19
	kBeforeExportReference : int = 20
	kAfterExportReference : int = 21
	kBeforeUnloadReference : int = 22
	kAfterUnloadReference : int = 23
	kBeforeSoftwareRender : int = 24
	kAfterSoftwareRender : int = 25
	kBeforeSoftwareFrameRender : int = 26
	kAfterSoftwareFrameRender : int = 27
	kSoftwareRenderInterrupted : int = 28
	kMayaInitialized : int = 29
	kMayaExiting : int = 30
	kBeforeNewCheck : int = 31
	kBeforeOpenCheck : int = 32
	kBeforeSaveCheck : int = 33
	kBeforeImportCheck : int = 34
	kBeforeExportCheck : int = 35
	kBeforeLoadReference : int = 36
	kAfterLoadReference : int = 37
	kBeforeLoadReferenceCheck : int = 38
	kBeforeReferenceCheck : int = 39
	kBeforeCreateReferenceCheck : int = 39
	kBeforePluginLoad : int = 40
	kAfterPluginLoad : int = 41
	kBeforePluginUnload : int = 42
	kAfterPluginUnload : int = 43
	kBeforeCreateReference : int = 44
	kAfterCreateReference : int = 45
	kExportStarted : int = 46
	kBeforeLoadReferenceAndRecordEdits : int = 47
	kAfterLoadReferenceAndRecordEdits : int = 48
	kBeforeCreateReferenceAndRecordEdits : int = 49
	kAfterCreateReferenceAndRecordEdits : int = 50
	kLast : int = 51
	pass
class MTimerMessage(MMessage):
	# addTimerCallback : staticmethod = <staticmethod object at 0x000002F149FB5DF0>
	__doc__ : str = "Class used to register callbacks for timer related messages."
	pass
class MUserEventMessage(MMessage):
	# registerUserEvent : staticmethod = <staticmethod object at 0x000002F149FB5E50>
	# isUserEvent : staticmethod = <staticmethod object at 0x000002F149FB5E80>
	# deregisterUserEvent : staticmethod = <staticmethod object at 0x000002F149FB5EB0>
	# postUserEvent : staticmethod = <staticmethod object at 0x000002F149FB5EE0>
	# addUserEventCallback : staticmethod = <staticmethod object at 0x000002F149FB5F10>
	__doc__ : str = "Class used to register callbacks for user event messages."
	pass
class MPaintMessage(MMessage):
	# addVertexColorCallback : staticmethod = <staticmethod object at 0x000002F143B80610>
	__doc__ : str = "Class used to register callbacks for paint related messages."
	pass
class MUiMessage(MMessage):
	# addUiDeletedCallback : staticmethod = <staticmethod object at 0x000002F143B8ACD0>
	# addCameraChangedCallback : staticmethod = <staticmethod object at 0x000002F143B8ACA0>
	# add3dViewDestroyMsgCallback : staticmethod = <staticmethod object at 0x000002F143B8A8E0>
	# add3dViewPreRenderMsgCallback : staticmethod = <staticmethod object at 0x000002F143B8A8B0>
	# add3dViewPostRenderMsgCallback : staticmethod = <staticmethod object at 0x000002F143B8A820>
	# add3dViewRendererChangedCallback : staticmethod = <staticmethod object at 0x000002F143B8A7F0>
	# add3dViewRenderOverrideChangedCallback : staticmethod = <staticmethod object at 0x000002F143B8A7C0>
	__doc__ : str = "Class used to register callbacks for UI related messages."
	pass
class MAnimMessage(MMessage):
	# addAnimCurveEditedCallback : staticmethod = <staticmethod object at 0x000002F16B0EA490>
	# addAnimKeyframeEditedCallback : staticmethod = <staticmethod object at 0x000002F16B0EA160>
	# addAnimKeyframeEditCheckCallback : staticmethod = <staticmethod object at 0x000002F16B0EA130>
	# addNodeAnimKeyframeEditedCallback : staticmethod = <staticmethod object at 0x000002F16B0EA100>
	# addPreBakeResultsCallback : staticmethod = <staticmethod object at 0x000002F16B0EA1C0>
	# addPostBakeResultsCallback : staticmethod = <staticmethod object at 0x000002F16B0EA190>
	# addDisableImplicitControlCallback : staticmethod = <staticmethod object at 0x000002F16B0EA040>
	# flushAnimKeyframeEditedCallbacks : staticmethod = <staticmethod object at 0x000002F16B0EA070>
	__doc__ : str = "Class used to register callbacks for anim related messages."
	pass

