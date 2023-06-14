

from tree.lib.object import TypeNamespace



class FileType(TypeNamespace):
	"""base class for file types - may not correspond exactly
	to file extensions
	"""

	class _Base(TypeNamespace.base()):
		name = ""

	class Json(_Base):
		name = "json"

	class Mesh(_Base):
		name = "mesh"

	class MeshSequence(_Base):
		name = "meshseq"

	class Image(_Base):
		"""any image format"""
		name = "img"

	class ImageSequence(_Base):
		"""images to be played in sequence, including video formats"""
		name = "imgseq"

	class USD(_Base):
		name = "usd"

	class Maya(_Base):
		name = "ma"

	class Houdini(_Base):
		name = "hip"

	class HDA(_Base):
		name = "hda"


