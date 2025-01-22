

#ifndef ED_DEFINES_H
#define ED_DEFINES_H 1


#define NAME( input, pt) ( point(input, "name", pt) )
#define ID( input, pt) ( point(input, "id", pt) )
#define SETPOINT(attr, pt, val) (setpointattrib(0, attr, pt, val))
#define SETPRIM(attr, pr, val) (setpointattrib(0, attr, pr, val))

// function modes
#define MEAN 0
#define MIN 1
#define MAX 2
#define RMS 3
#define SUM 4
#define PRODUCT 5

// i have tasted of the macros as leto tasted of the spice
// i now undergo metamorphosis, as leto underwent metamorphosis

#define STR(A) A#
#define S_CAT_(A,B) A##B
#define S_CAT(A,B) S_CAT_(A,B)
// whosoever dreamed of such a stupid question as "why?"

// define function aliases to allow efficient dispatching by element type
#define nprims nprimitives

#define nvertexs nvertices



#define ELT point

// #if strcmp(ELT, vertex)
//      #define NELEMENTS nvertices
// #else
//
// #endif

// WHAT I WANT:
/*
#define TESTVAL one
#define CONDITION_ STR(TESTVAL) == one) ? "true" : "false"
#define CONDITION STR(CONDITION_)
*/
// be able to test a defined value WITHIN a define, and change its output string


#define NELEMENTS S_CAT(n, S_CAT(ELT, s))

#define FNS(VT) \
    function VT[] STR(ELT)_vals(int geo; string atname){ \
        int nels = NELEMENTS(geo);\
        VT vals[]; \
        resize(vals, nels); \
        for(int i = 0; i < nels; i++){ \
            append(vals, VT(point(geo, atname, i))); \
        }\
        return vals; \
    }\


//FNS(float)


#define ELTFNS \
    FNS(float) \
    FNS(int) \
    FNS(vector) \
    FNS(vector4) \
    FNS(matrix) \
    FNS(matrix2) \
    FNS(matrix3) \
    FNS(string) \

#undef ELT

#define ELT prim
ELTFNS
#undef ELT

#define ELT point
ELTFNS
#undef ELT

#define ELT vertex
ELTFNS
#undef ELT


/* seems very basic but controlling separate #defs is easier than doing logic
 inside the macros
 #define ELT point

 #if strcmp(ELT, vertex)
     #define NELEMENTS nvertices
 #else
     #define NELEMENTS CAT(N, CAT(ELT, S))
 #endif

 #define FNS(VT) \
     function VT[] STR(ELT)_vals(int geo; string atname){ \
     int nels = NELEMENTS(geo); \
     VT vals[]; \
     resize(vals, nels); \
     for(int i = 0; i < nels; i++){ \
         append(vals, VT(point(geo, atname, i))); \
     }\
     return vals; \
 }\
 //

 #define VALTYPE\
     FNS(vector)\
     //

 VALTYPE
*/



#endif
