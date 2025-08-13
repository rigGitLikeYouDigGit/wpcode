from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""test for decoupling topology from resolution
with ephemeral patches driving points, reprojecting live 
to their previous shape when moved

should be hierarchical - points and lines declared in the space
of a parent patch should move when 
deformed
final topology can be strung between features of any depth

spirals are illegal


upper level of the head would just be a box, front ridge as brow,
lower as jaw?

ignore speed for now - if this is a valid way to deform meshes,
i'm all about it


might change the name to Strata


this is now rewritten in maya/plugin/strata
"""
