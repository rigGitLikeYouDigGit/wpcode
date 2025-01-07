
#ifndef ED_MESHMATRIX_H
#define ED_MESHMATRIX_H
// somewhat unhinged tests for operating on sparse NxN matrices
// as literal poly meshes
#include "ed_maths.h"
#include "ed_poly.h"


int addmatrixpoint(int row, column, origindex, ngeopts; float matscale){
    // create new matrix points in a square according to matscale
    // ngeopts is number of points in geo to matrix-ify
    vector pos = set(row, 0, column);
    pos *= matscale / ngeopts;
    int npt = addpoint(0, pos);
    setpointgroup(0, "matpts", npt, 1);
    setpointattrib(0, "row", npt, row);
    setpointattrib(0, "column", npt, column);
    setpointattrib(0, "origindex", npt, origindex);
    setpointattrib(0, "matindex", npt, row * ngeopts + column);
    setdetailattrib(0, "nrows", ngeopts);
    return npt;
}

int getmatpoint(int geo, row, column){
    return (int(detail(geo, "nrows")) * row + column);
}

float getmatvalue(int geo, row, column; string attr){
    return point(geo, attr, getmatpoint(geo, row, column));
}

function float[] getmatrowpts(int geo, row){
    // assumes only row prims
    return (primpoints(geo, pointprims(geo, getmatpoint(geo, row, 0))[0]));
}

int getmatdiagonalpt(int geo, index){
    return getmatpoint(geo, index, index);
}



#endif
