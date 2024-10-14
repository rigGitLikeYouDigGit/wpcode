
"""there's just something about nested structures that
pulls me in like an addict

refit to work with wpDex objects as the "model" spine,
then all the Qt rubbish hooked into it

for hierarchy we chain SuperItem -> SuperModel -> SuperItem -> SuperModel,
each level only knowing about its immediate children

later, consider each view defining vertical and horizontal modes
add option to collapse levels, use type labels to shorthand what
levels contain, etc

line widget for showing path to item, and for listing / collapsing

text option for editing full structure as json / serialised string


atomic widget is more specialised still, literally just exists for consistent set/get value interface for stock qt widgets like text and checkbox
"""

from .base import WpDexWidget, WpDexWindow

from .seqitem import SeqDexWidget
from .primitem import PrimDexWidget
from .stritem import StrDexWidget

from .react import ReactiveWidget, WidgetHook
from .atomic import *