
from __future__ import annotations
import typing as T

"""lib for wpdex - doesn't stand for anything, just a name

this is our overall pathing / expression resolution system,
and associated ui tools. 

depends on wplib and wpui

should this also handle reactive updates, links to UI etc?
the dream for me is to have a single model across the entire
program, updating live whenever a change comes from code or ui - 
obviously that's the whole point of QT, but the default views
are not enough 


"""


from .base import WpDex


from .primdex import *
from .dictdex import *
from .seqdex import *
from .strexpdex import *

