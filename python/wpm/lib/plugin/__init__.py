
"""for safety, importing openmaya directly within this package -
it might be fine to use the wp version"""

from wpm.constant import WP_PLUGIN_NAMESPACE
from .template import PluginNodeTemplate, PluginDrawOverrideTemplate, NodeParamData, PluginNodeIdData
from .helper import MayaPluginAid, MayaPyPluginAid

# studio-specific derived class for plugin aid -
# subclass and override studioIdPrefix() to return the prefix for your own studio


class WpPyPluginAid(MayaPyPluginAid):
	@classmethod
	def studioIdPrefix(cls)->int:
		"""return the prefix for studio plugin ids,
		used as a namespace for plugin node ids"""
		return WP_PLUGIN_NAMESPACE

#TODO: refactor this to the WPROOT system
wpPluginAid = MayaPluginAid(
	"wpPlugin",
	"C:/Users/ed/Documents/GitHub/wpcode/maya/plugin/out/build/x64-Debug/wpplugin.mll"
)

