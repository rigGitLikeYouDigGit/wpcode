bl_info = {
	"name": "WP_test",
	"blender": (2, 80, 0),
	"category": "Object",
}
def register():
	print("Hello World plugin")
def unregister():
	print("Goodbye World plugin")

if __name__ == '__main__':
	register()
