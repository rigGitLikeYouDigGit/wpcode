

"""

scripts to populate "gen" folder
with generated node classes for every node type in maya



combine with lazy loading / dynamic import system

each node has a generated class in "gen",
optionally inherited by a user-defined class in outer folder

BUT, subclasses of a node extended by a user still need
to inherit from it

"""