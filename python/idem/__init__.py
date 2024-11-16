
"""
overall launcher for asset-focused program sessions

maybe replace with allzpark / rez after we get everything working,
if it can be made to work with the pipeline system

we could go super hard here in setting up plugin interfaces, registering
packages to run any random code on startup -
not yet.

just get it to launch the programs and add the WP code paths

STRUCTURE :
anything in top-level is abstract base classes and/or pure python, that can run
	directly in the standalone Idem program

".dcc" package holds outwards-facing wrapper objects to handle program sessions -
	finding windows, getting processes, startup, and actually sending commands to
	start up an idem process in a dcc session live (if the program can support it)

we define a separate package for each supported dcc, where we specialise the
relevant classes and chimaera nodes
each dcc-specific package may import domain-specific code

TODO: a better name
 ilex
 concourse
 concord (rip)
 wrangle
 wipe
 wot
 tangle
 tara
 tarana
 brunel
 isle?
 isles?

"""

from . import adaptor, model, node

from .maya import *