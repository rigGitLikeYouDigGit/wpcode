from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import subprocess
import re


def get_changed_line_ranges(commit1, commit2, repo_path="."):
	"""
	Returns a dictionary of { 'filepath': [(start_line, end_line), ...] }
	representing the added and modified line ranges in the destination commit.
	"""

	# We use -c overrides to guarantee uniform output:
	# 1. core.quotePath=false prevents git from quoting file paths with special chars
	# 2. diff.noprefix=false guarantees the standard 'a/' and 'b/' prefixes are present
	cmd = [
		"git",
		"-c", "core.quotePath=false",
		"-c", "diff.noprefix=false",
		"diff",
		"-U0",  # 0 lines of context (isolates exact changes)
		commit1,
		commit2
	]

	# Execute the Git command
	result = subprocess.run(
		cmd,
		cwd=repo_path,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		check=True
	)

	changes = {}
	current_file = None

	# Regex to extract starting line and length from unified diff chunk headers
	# Example matched: @@ -53,4 +55,2 @@ def some_function():
	# Group 1 (new start line) = 55, Group 2 (new length) = 2
	chunk_regex = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

	for line in result.stdout.splitlines():

		# 1. Identify the file being patched
		if line.startswith("+++ "):
			if line.startswith("+++ /dev/null"):
				current_file = None  # File was entirely deleted, ignore it
			else:
				# Strip the '+++ b/' (first 6 characters) to get the clean file path
				current_file = line[6:]
				if current_file not in changes:
					changes[current_file] = []
			continue

		# 2. Identify the changed line blocks for the current file
		if current_file:
			match = chunk_regex.match(line)
			if match:
				start_line = int(match.group(1))
				length_str = match.group(2)

				# If length is omitted in diff, it implies exactly 1 line changed
				if length_str is None:
					length = 1
				else:
					length = int(length_str)

				# A length of 0 means lines were *only* deleted.
				# Because they don't exist in the target commit, we skip them.
				if length == 0:
					continue

				# Calculate the inclusive end line
				end_line = start_line + length - 1
				changes[current_file].append((start_line, end_line))

	# Clean up files that ended up with no lines (e.g., pure deletions)
	return {f: ranges for f, ranges in changes.items() if ranges}


# --- Example Usage ---
if __name__ == "__main__":
	# Compare the previous commit (HEAD~1) with the current commit (HEAD)
	# You can substitute this with commit SHAs, branch names, etc.
	changes = get_changed_line_ranges("HEAD~1", "HEAD")

	import pprint

	pprint.pprint(changes)

