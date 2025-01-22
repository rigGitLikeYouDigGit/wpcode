from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""tests for ways to watch for changes within maya - 
for example, the coveted live-link sets / hierarchy

we first try a simpler async "pull" based, where a timer
checks watched nodes / files for changes, and propagates them
when detected

we can actually achieve this most easily without doing callbacks
at all - a callback registered for maya's idle event will trigger
infinitely, so has to be continuously registered and deregistered

"""




