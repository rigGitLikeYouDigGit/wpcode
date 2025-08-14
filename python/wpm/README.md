# wpm :
maya package


## core
- deep integration with api, plug and node systems

- `getMFn()` : returns the best-fitting api MFn class for the given MObject - eg if you pass a transform, you'll get an MFnTransform, not just an MFnDagNode.
  - Best-fitting relations are cached against numeric API type once found, so the whole system gets faster the more it's used.

### `core.Plug`
- Treating Maya plugs with our shared TreeInterface
  - Allows pathing, selecting with string values, broadcasting plugs and plug connections. Probably also serialisation?
- `use()` : replacement for `cmds.setAttr` and `cmds.connectAttr` at the same time
  - can also say `myNode.attribute_ = myOtherNode.attribute_` to make a connection directly
  - `myNode.attribute_ = myOtherNode.attribute_()` to set a static value
- connection operations can all be accelerated by passing in an MDGModifier, so you can set up a block of rig connections and then execute them only once
- `PlugDescriptor` on a node class or plug is what gives type hinting and autocompletion of attributes in PyCharm, so every attribute of every node is its own custom type

### core.node
 - `WN` : Base class and single import for all custom node wrappers
   - Name is short to maximise gains over `cmds.createNode`  
   - `_codegen`
     - Requirements:
       - We need to generate maya classes for every Maya node with PlugDescriptors for attribute hints
       - We need to hand-write custom overrides for certain types of nodes - DagNode, Transform etc - **without** those getting blasted when we rerun code generation.
     - Solution:
       - **Zigzag inheritance** between authored and generated classes (if you imagine the two as different sides of a wall)
       - `gen.DagNode( Entity )` is the generated class for DagNode, you can see it in gen/dagNode.py. `Entity` is found with a dynamic lookup `retriever.getNodeCls("Entity")` when the file is imported.
       - `author.DagNode( gen.DagNode )` is the authored override class, inheriting from the generated DagNode
       - `gen.Transform( DagNode )` is the generated class for Transform; again, DagNode is found with `retriever.getNodeCls("DagNode")`. **This checks first for an author file, and then a generated file**
       - `author.Transform( gen.Transform )` - again, we can override specific Transform functionality at will.
     - I don't care if it's complicated, this is rad :D
   - All Maya node classes accessible as class attributes - `WN.Transform`, `WN.NurbsCurve` etc 
   - Intercepts the call on initialisation to return the right type of node wrapper for the given node, eg:
     - `n = WN("pCube1")`
     - `type(n)` -> WN.Transform

   - You may wonder, "How does a base class have its own child classes as class attributes?"
     - All the generated and authored derived classes are only imported at type checking time (eg not in execution), then dynamically found and cached on request. Importing all those thousands of types at once would make you beg for pymel's speed
 - No global caching for plug or node instances - they each hold only an MObject or MPlug respectively

### core.cache
- for ease of use / sanity, we generate full copy of the entire OpenMaya2 api and cache it here. 
  - This is to pre-build things like best-fitting MFns, but also wrap all functions to accept WNs and Plug trees, but also more easily accept things that turn to strings or MObjects. Still more to do here. 
  - I didn't really know the best way to tackle this, so between `cache` and `patch` there's a bit of sweater spaghetti with which file holds what, which file gets imported etc. Could be better.

## lib
Files focusing on specific areas

### `lib.scene.SceneDelta`
- maybe THE most useful, most BASIC thing I've ever made - context manager to take a difference in maya node names before and after a code block. 
  - No longer need to worry about capturing all the bonkers stuff from an fbx import, or a rig build, or one of the cmds operations that makes a ton of underworld nodes 

### `lib.plugin.helper.MayaPluginAid`
- Useful helper class to manage the loaded state of a plugin, either in python or c++ - also includes function to update generated node classes from new plugin nodes

## chimaera
- Chimaera execution nodes for bringing in models and simple rigging operations

## resource
- Mainly shader files for the GLSLShader node