
if __name__ == '__main__':
	import sys, os
	from pathlib import Path

	WP_ROOT = os.getenv("WEPRESENT_ROOT")
	assert WP_ROOT, "WEPRESENT_ROOT not set in environment"
	WP_ROOT_PATH = Path(WP_ROOT)
	WP_PY_ROOT = WP_ROOT_PATH / "code" / "python"
	print("root", WP_PY_ROOT)
	sys.path.insert(0, str(WP_PY_ROOT))
	import idem
	from idem.ui.window import Window

	Window.launch()
