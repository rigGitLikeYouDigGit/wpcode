
from __future__ import annotations
import typing as T

"""
Redoing serialisation framework - it was already pretty decent,
but this has to be absolutely foolproof.

There are several issues to separate and solve:
- Serialise references to user-defined object types in code, which may move around.

- Serialise external or STL objects, where types cannot be modified.

- Serialise / deserialise arbitrarily nested data structures with single top-level call.

- Serialise data in future-proof way, so that a more modern version of the same type
can still load old serialised data.


adapters / subclasses to specify UID for serialisation.


Provide separate function or tool to run over a pool of serialised data files,
reading back which data versions of which types are actually in use.


serial data is always a dict, with a single key, "d", holding the data,
"f" holding the format data, and "v" holding the version data.
d is directly returned by encode() and passed to decode().
- this costs in space and readability, but makes it easier to handle,
a totally uniform data structure.

more files than needed in this module to give more descriptive tracebacks


to catch instances where types might have moved around, consider either tracking
or a full first pass over structure to find and load all types needed -
then check which ones aren't found and flag them

can then pass this to a UI, or pass a function hook directly to resolve them
THEN run full deserialisation

if an object can't be deserialised, consider an overall "UNKNOWN" type to preserve
data in a read-only state in a tool, even if it can't be directly loaded

"""

from .adaptor import SerialAdaptor

# import and register all default types
from . import applied as _applied

# import main functions
from .main import serialise, deserialise

# import the custom base class
from .abc import Serialisable, SerialisableAdaptor

