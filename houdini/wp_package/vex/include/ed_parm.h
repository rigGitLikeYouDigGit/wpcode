
#ifndef ED_PARM_H
#define ED_PARM_H

#include "ed_array.h"

/* tests to allow overriding wrangle channels with attributes, if present 
top level : 

ATCHF(myval, 0, atparmname, @ptnum)
syntax is a bit weird

ATCH(float, myval, 0, atparmname, @ptnum) ?

this would be the dream:

float myval = ATCHF(0, atparmname, @ptnum);

but right now I can't see how to capture the var to assign to,
unless it's passed to the macro
*/


// Define type-specific channel functions
#define _TYPECH_float chf
#define _TYPECH_vector chv

// Define type-specific getattrib wrappers
#define _ATCH_FN(type, chfn) \
type _get##type##attrorparm(int geo; string name; int elemnum) { \
    int success; \
    type v = getattrib(geo, "point", name, elemnum, success); \
    if (!success) { \
        v = getattrib(geo, "prim", name, elemnum, success); \
        if (!success) { \
            v = chfn(name); \
        } \
    } \
    return v; \
}

// Instantiate for float and vector
_ATCH_FN(float, chf)
_ATCH_FN(vector, chv)

// Main macro - uses token pasting to call the right function
#define ATCH(type, var_name, geo, at_parm_name, el) \
	type var_name = _TYPECH_##type##(at_parm_name);\
    var_name = _get##type##attrorparm(geo, at_parm_name, el);

#endif