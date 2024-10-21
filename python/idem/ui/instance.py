

from __future__ import annotations
import typing as T


from PySide2 import QtCore, QtWidgets, QtGui

from wp.pipe.ui import AssetSelectorWidget

"""overall window representing single instance of Idem - 


TODO: 
    come back and rewrite this properly with fields and generated widgets
    
    
we might create the path widget thus - 

path = PathWidget(parent=None,
	default="strategy.chi",
	parentDir=lambda : ref("asset").diskPath(),
	postProcessFn=lambda rawPath : Path(rawPath).relativeTo(self.parentDir())
    
"""



