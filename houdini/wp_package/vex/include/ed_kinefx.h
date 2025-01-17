
#ifndef ED_KINEFX_H
#include "kinefx.h"

int isjoint(string geo; int pt){
    // does point have name and localtransform attributes
    int success;
    return (len(string(getattrib(geo, "point", "name", pt, success))) != 0);
}

int isjoint(int geo; int pt){
    return isjoint(INPUT(geo), pt);
}

// int getparent(string geo; int pt)
// {
//     int n_pts[] = neighbours(geo, pt);
//     foreach(int npt; n_pts)
//     {
//         int hedge = pointhedge(geo, npt, pt);
//         if(hedge > -1)
//         {
//             return npt;
//         }
//     }
//     return -1;
// }
//
// int getparent(int geo; int pt)
// {
//     return getparent(INPUT(geo), pt);
// }
//

int getroot(string geo; int pt){
    //// SO ////
    /* reminder to copy any input points, since in snippets
    they are passed by ref, and mess up the whole node */
    int localpt = pt;
    if(isjoint(geo, localpt) == 0){
        return -1;
    }
    int parent = getparent(geo, localpt);
    while(parent != -1 ){
        localpt = parent;
        parent = getparent(geo, localpt);
    }
    return localpt;
}
int getroot(int geo; int pt){
    return getroot(INPUT(geo), pt);
}
#define ED_KINEFX_H
#endif
