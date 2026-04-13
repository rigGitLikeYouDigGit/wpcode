
#ifndef ED_LANDMARK_BOX
#define ED_LANDMARK_BOX 1

/* cube/bounding box landmark functions
orienting, extruding, get vectors for rounding etc


on creation, box will have src/dst faces (in xyz these would be
base/top), and up (in xyz by default x)

we get an indirection of face indices to map these important
ones to constant indices?
or we assume each box will have "lboxsrc", "lboxdst" etc attributes?
probably better



*/


#define LBOX_SRC 0
#define LBOX_DST 1
#define LBOX_UP 2




#endif


