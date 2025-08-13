# wp
 
Personal codebase for rigging and VFX. Covers Maya, Houdini, Qt and pure python.


# Highlights

# Strata
Strata is a bit mental.
- Logline: Instead of **rigging** a **sculpted** model, we **sculpt** a **rigged** model.
  - We first build a topology manifold out of points (matrices), edges (bezier curves) and faces (bezier patches).
    - Detail is added in layers of geometry, hence "Strata" - each is parametric in the space of its drivers.
  - We tessellate that manifold to polygons and pass it to the modeller as a base mesh - they sculpt the base shape, face shapes, correctives etc and pass them out as separate meshes.
  - **Backpropagation** then best-fits our original manifold graph to the sculpted targets, with final deltas being parametric offsets in the space of drivers.
    - Blending between face shapes is now a holistic blend between large volumes and fine details, all at once, much closer to the real volumes of the face.
    - In cartoon characters, smooth deformation is inherited by every detail.
    - All curves and surfaces remain live underneath displacement - we basically get Pixar's Curvenet for free.
  - Changes to the base topology are now much easier - to add a limb to a human biped, literally append that manifold on top of the original, change the final meshing, and author only the textures / weights / displacements around the join. All the original data is still there, just hidden.
- **This at least is the dream**, still a massive amount of work to do.
- Implemented in `maya/plugin/strata` as a pure c++ graph, with integrations for maya nodes. Intending to expose as much as possible to MPxCommands, to handle UI in Python
  - Similar concept to APEX, each Maya node holds a new, modified copy of the whole Strata graph, and only the final graph is evaluated / trained with back-propagation.
- Technically also includes a freely-typed interpreted language for topological expressions.


rest of the code below is in `/python/`
## Maya
`wpm` : Maya libraries
- `wpm.WN` : All-singing all-dancing maya node wrapper. Heavy focus on fluid and readable syntax. Numpy-esque plug value broadcasting. Like PyMEL but better.
  - `wpm.core.node._codegen` : TYPE HINTING for EVERY ATTRIBUTE of EVERY NODE in Maya with generated classes -
  - Zigzag inheritance between authored and generated classes allows manual overrides at any level of maya's node taxonomy, in harmony with re-running code generation.
  - I committed the generated classes for discussion at `wpm.core.node.gen` but normally there's no point versioning them
- `wpm.lib.skin` : Convert skin weights to numpy arrays - needs refactor to scipy sparse types
- `wpm/resource/shader` : Various shader files for Maya's glslShader, including the eyeShader and 3dPrint / volumetric slice shader from demo reel.
- `wpm.tool.eph` : Old work towards Raf Anzovin's Ephemeral Rig idea - I absolutely love this, need to get back to it one day
- `wpm.tool.feldspar` : Test for a rigid solver plugin using Rigidity Theory, closely following this project https://github.com/hrldcpr/linkages - can do a simple 4-bar, but I couldn't work out how to do more complex constraints.

## Houdini
`wph` : Houdini python functions
 - Should work with `\houdini\wp_package` as a drag-drop houdini package
 - `\houdini\wp_package\vex` Vex libraries focusing on semantic data with geometry, parallel transport, matrix operations, etc

## Blender
`wpblend` : the bare minimum to get sculpts out of blender

## Python
`idem` : Clients and server to sync arbitrary scene data between arbitrary DCCs. Nvidia Omniverse without being bound to USD (and Nvidia). Inspired by Dreamworks' Arras
 - Proof of concept only, got it working with cameras between Maya and Houdini 

`wplib` : general python libraries
 - `wplib.object` : Mixins and base classes - Pathable, Adaptor, Visitor, TypeNamespace
 - `wplib.serial` : Low-pain classes and adaptors to human-readably serialise any arbitrary python structures.
 - `wplib.inheritance` : Inspect bases, register overrides against base classes

`wptree` : Tree-like python data structure and mixin interface, each branch having name, value and children

`wpexp` : Custom expression evaluation and syntax rules toolkit.

`wpdex` : Metadata and deep mapping of arbitrary python structures
- Automatic display and editing in ui.
- Visitor integration to find and evaluate expressions / references, set overrides and validation rules by string paths
- Track deltas to structure, giving automatic undo/redo
- ...which all sounds great, but it's UNUSABLY slow, and when it breaks (which is frequent) the tracebacks are incomprehensible.
  - todo: make tracker asynchronous

`wpui` : Qt utilities and library widgets
- `wpui.canvas` : Makes it *slightly* less painful to work with graphics scenes and make new node graph viewports in tools, with adaptors and mixins.


# Lowlights

`wplib.object.to()`/`ToType()` : meme object, builds a networkx directed graph of type conversions between objects, and caches the path of functions on request. Useful for type coercion from function type hints.
- eg if someone passes an MTransformationMatrix but you want a scipy.spatial.transform.Rotation, conversion will automatically route through (MTransformationMatrix -> MMatrix -> numpy.ndarray -> scipy...Rotation)
 - "I asked for an MFloatVector, now give me an MFloatVector"

`wplib.inheritance._MetaResolver` : Makes QtWidgets work nicely with multiple bases and metaclasses - intercept type creation to generate a new metaclass with metaclasses of all the given bases as bases *of that metaclass*. Sorry Guido.

`wpm.tool.keyboard.piano` : have a piano

`chimaera` : A dependency graph dynamic enough to emulate itself.
- Intended to be the base for any kind of execution in pipeline - rig build, scene build for rendering, ephemeral rig evaluation etc.
- Evaluation, node class lookups, domain-specific node packages for maya, inheriting etc all works
- Needs to be simplified. 
- I was interested in a system where every part of it could be linked, could be resolved dynamically - something like creating your own syntax with lisp macros, but within an evaluation framework, a responsive/living tool. Chimaera explores that but it doesn't succeed.
- Never fully resolved the APEX question - does it build a graph, or does it build a graph that builds a graph
