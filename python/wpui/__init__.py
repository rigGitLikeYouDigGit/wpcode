"""if necessary this package can depend on wptree,
but wptree.ui will depend on this"""

# import and register adaptors, don't directly expose them
from . import adaptor as _
#from .theme import STYLE_SHEET


"""
TODO:
- extensible event filter object? not worth it yet but might be later

"""