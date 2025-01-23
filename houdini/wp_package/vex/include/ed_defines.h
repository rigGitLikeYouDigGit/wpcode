

#ifndef ED_DEFINES_H
#define ED_DEFINES_H 1


#define NAME( input, pt) ( point(input, "name", pt) )
#define ID( input, pt) ( point(input, "id", pt) )
#define SETPOINT(attr, pt, val) (setpointattrib(0, attr, pt, val))
#define SETPRIM(attr, pr, val) (setpointattrib(0, attr, pr, val))

// function modes
#define ED_MEAN 0
#define ED_MIN 1
#define ED_MAX 2
#define ED_RMS 3
#define ED_SUM 4
#define ED_PRODUCT 5

// modes for working with certain element types
#define PRIMT 0
#define POINTT 1
#define VERTEXT 2


// i have tasted of the macros as leto tasted of the spice
// i now undergo metamorphosis, as leto underwent metamorphosis

#define STR(A) A#
#define S_CAT_(A,B) A##B
#define S_CAT(A,B) S_CAT_(A,B)
// whosoever dreamed of such a stupid question as "why?"

// define function aliases to allow efficient dispatching by element type
#define nprims nprimitives
#define nvertexs nvertices


string elname(int mode){
    if(mode == PRIMT) return "prim";
    if(mode == POINTT) return "point";
    //if(mode == VERTEXT)
    return "vertex";
}
int nelements(int mode, geo){
    if(mode == PRIMT) return nprims(geo);
    if(mode == POINTT) return npoints(geo);
    //if(mode == VERTEXT)
    return nvertices(geo);
}



#define ELT point

#define NELEMENTS S_CAT(n, S_CAT(ELT, s))

#define FNS(VT) \
    function VT[] STR(ELT)_vals(int geo; string atname){ \
        int nels = NELEMENTS(geo);\
        VT vals[]; \
        resize(vals, nels); \
        for(int i = 0; i < nels; i++){ \
            vals[i] = STR(ELT)(geo, atname, i); \
        }\
        return vals; \
    }\
    function VT[] STR(ELT)_vals(int geo; string atname; int ids[]){ \
        int nels = len(ids);\
        VT vals[]; \
        resize(vals, nels); \
        foreach(int id; ids){ \
            append(vals, VT(point(geo, atname, id))); \
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

// template only by value type
#define TYPEFNS(VT) \
    function VT combine( VT vals[]; int mode ){ \
        VT result = VT(); \
        if(mode == ED_MAX){\
        }\
        return result; \
    }\
    function void setelval(int mode; int geo; string atname; int el; VT val){\
        if(mode == PRIMT) setprimattrib(geo, atname, el, val);\
        if(mode == POINTT) setpointattrib(geo, atname, el, val);\
    }\
    function void setelvals(int mode; int geo; string atname; int els[]; VT vals[]){\
        foreach(int i; int el; els){\
            setelval(mode, geo, atname, el, vals[i]);\
        }\
    }\
    function void setelvals(int mode; int geo; string atname; int els[]; VT val){\
        foreach(int i; int el; els){\
            setelval(mode, geo, atname, el, val);\
        }\
    }\
    function void setelvals(int mode; int geo; string atname; VT val){\
        for(int i= 0; i < nelements(mode, geo); i++){\
            setelval(mode, geo, atname, i, val);\
        }\
    }\
    function VT getelval(int mode, geo; string atname; int idx){\
        if(mode == PRIMT) return prim(geo, atname, idx);\
        return point(geo, atname, idx);\
    }\
    function VT[] getelvals(int mode, geo; string atname; int els[]){\
        VT vals[]; \
        resize(vals, len(els)); \
        foreach(int i; int el; els){ \
            vals[i] = getelval(mode, geo, atname, el); \
        }\
        return vals; \
    }\
    function VT[] getelvals(int mode, geo; string atname){ \
        int nels = nelements(mode, geo);\
        VT vals[]; \
        resize(vals, nels); \
        for(int i = 0; i < nels; i++){ \
            vals[i] = getelval(mode, geo, atname, i); \
        }\
        return vals; \
    }\



TYPEFNS(float)
TYPEFNS(int)
TYPEFNS(vector)
TYPEFNS(vector4)
TYPEFNS(matrix)
TYPEFNS(matrix2)
TYPEFNS(matrix3)
TYPEFNS(string)

function matrix product(matrix mats[]){
    matrix base = matrix();
    foreach(matrix mat; mats){
        base = base * mat;
    }
    return base;
}

#endif
