from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""just some thoughts - 
for sculpting something like an anatomical character, there are
many different pieces that all affect each other - skeleton, muscles, fascia, fat, skin, armature, controls, sculpted shapes, etc

would it be useful to show that somehow, and let all those relationships be
live in whatever software is best to affect them?

premise:
work on the DELTAS made by hand to each entity, and propagate them all out
to the other pieces as they happen.

this might mean a full NxN network, where each directed connection could be a custom geometry operation

"""


