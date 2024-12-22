
from __future__ import annotations
import typing as T, types

"""
for rig structure, we assume that bits of our puppet can be built and
regenerated on the fly, but a deformation rig needs consistent structure
to be feasible - we can't repaint weights for every different control scheme

control -> armature -> deformation rig

micro-rig flash-rig kata-rig
anti-rig?

simplest armature is obviously a skeleton, but this also
includes curves, techmeshes, lattices etc - 
every hook that a micro-puppet can drive should be present in the armature - 
this will of course get silly if not properly managed
coupling components???? distant framestore riggers screech in rage

the micro-rig should only be for large-scale kinematic problems - deformation rig
can still include anim controls and details, and the easiest solution is just to
do some animation the normal way

armature is also what final keys are applied to and loaded from, 
after ephemeral effects applied - it's what we use to blend poses, train and
apply mocap, and so on

armature includes all dynamics settings and constraints? coliders, soft bodies
required if we allow sculpting in key frames
MAYBE NOT, where is line between this and full muscle volumes on body

does armature relate to meshes
no solid coupling?
OPTIONAL coupling? benefit for fingers, limbs etc to define consistent topology
maybe we just give up on the dream of instancing fingers, it's never
worked out

building body rig and anatomy meshes can dump bits of armature together? direction
of armature hierarchy not so important


doesn't matter where bits of the armature come from - but cannot be BUILT from puppet?
ironically puppet is end of chain in build, front in evaluation?
i guess joint pivots do drive control pivots 


COMMAND -> ARMATURE -> CLAY


in building, final deformation and action of clay drives armature, which
drives the positions of the command rigs

in evaluation, command rigs drive the armature, which drives the final deformation

wow controls drive joints and joints drive deformation,
very revolutionary and new, nice one

fair, but key dev is that we STORE dense keys on the armatures - for joint orients,
curve cvs, techmesh vertices? we get into difficulties on updating and changing topo,
but the point stands - 
this really isn't for anything below kinematic blocking,
clay rig can define sculpts, 



"""

from .base import MayaOp

# abbreviate headings in chimaera tree summary
{"cm" : {},
 "am" : {},
 "al" : {}
 }
from .node import skeletonop
#from . import *

"""
example of head

import face mesh,
import skull mesh,
define pivots for skull and jaw - armature



"""





