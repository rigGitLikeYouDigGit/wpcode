
"""
I first started trying to make this ui widget about 5 years ago -
some versions have been better than others.

For this one, use Qt better -

single root tree has a single Qt model object associated with it by uid -
model rebuilds itself from tree whenever tree changes,
and model serves only to interact with uis and interpret ui signals and events.

complex tree drawing is handled entirely in delegate class overrides -
if we have to do some loopy stuff in the delegates to take power away from
view, that's ok.

View widget is relatively thin, serves to inject the types of delegates to use
in drawing. Passes all signals to model, has no direct interaction with tree.
Multiple views can be attached to a single model.

Actions should probably be provided by items - items and delegates should
be the only classes that a user needs to override

"""

from wptree import Tree, TreeInterface

from wpui.superitem import SuperItem

from .model import TreeModel
from .item import TreeBranchItem
from .view import TreeView

from .superitem import TreeSuperItem

SuperItem.registerPlugin(TreeSuperItem, TreeSuperItem.forCls)
TreeSuperItem.modelCls = TreeModel

