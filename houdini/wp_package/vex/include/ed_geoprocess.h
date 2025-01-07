
#ifndef ED_GEOPROCESS_H
#define ED_GEOPROCESS_H
// functions for proper geometry processing
// assume trimesh unless otherwise specified
#include "ed_maths.h"
#include "ed_poly.h"


function float vertexangle(int geo, vtx){
    // angle between edges incident to this vertex
    vector vtxpos = point(geo, "P", vertexpoint(geo, vtx));
    vector prevvtxpos = point(geo, "P",
        hedge_presrcpoint(geo, vertexhedge(geo, vtx)));
    vector nextvtxpos = point(geo, "P",
        hedge_dstpoint(geo, vertexhedge(geo, vtx)));
    float costheta = (dot(
        normalize(prevvtxpos - vtxpos), normalize(nextvtxpos - vtxpos)
    ));
    return acos(costheta);
}


function float pointangleflatness(int geo, pt; string vertexangleattr){
    /* return measure of point curvature, as
    2pi - (all its edge angles)
    */
    float angles[];
    foreach(int vtx; pointvertices(geo, pt)){
        append(angles, float(vertex(geo, vertexangleattr, vtx)));
    }
    return 2*PI - sum(angles);
}

function float laplacianoperator(
    int geo, ptnum;
    string pointvalueattr, // name of value attribute for laplacian
    pointthirdareaattr,
    vertexcotanangleattr

){
    float neighbourvalues[];
    float centralval = point(geo, pointvalueattr, ptnum);
    float centralthirdangle = point(geo, pointthirdareaattr, ptnum);
    foreach(int npt; neighbours(geo, ptnum)){
        // get vertices opposite to either side of hedge connecting points
        int linkhedge = pointhedge(geo, ptnum, npt);
        float neighval = point(geo, pointvalueattr, npt);
        float anglea = vertex(geo, vertexcotanangleattr,
            hedge_postdstvertex(geo, linkhedge));

        float angleb = vertex(geo, vertexcotanangleattr,
            hedge_postdstvertex(geo,
                hedge_nextequiv(geo, linkhedge)));

        append(neighbourvalues,
            (anglea + angleb) * (neighval - centralval)
        );
    }
    return (1.0 / (2 * centralthirdangle) * sum(neighbourvalues));
}


function float[] cotanweights(int geo; int ptnum){
    // return array of cotan laplacian weights for this point
    // unsorted but order is consistent
    // LAST VALUE is combined prim area weight of point
    // requires prim attribute "area"
    // iterate over vertices, get hedge, then get hedge equiv

    float weights[];
    int basevtx = pointvertex(geo, ptnum);
    int ahedge, bhedge, aprim, bprim;
    vector thispos, neighpos, aptpos, bptpos;
    thispos = point(geo, "P", ptnum);
    float acotan, bcotan;
    float areasum = 0.0;
    float cotsum = 0.0;
    foreach(int vtx; woundpointvertices(geo, basevtx)){
        ahedge = vtxhedge(geo, vtx);
        aprim = hedge_prim(geo, ahedge);
        neighpos = point(geo, "P", hedge_dstpoint(geo, ahedge));
        aptpos = point(geo, "P", hedge_presrcpoint(geo, ahedge));
        acotan = 1.0 / tan(acos(dot(aptpos - thispos, aptpos - neighpos)));

        bhedge = hedge_nextequiv(geo, ahedge);
        bprim = hedge_prim(geo, bhedge);
        bptpos = point(geo, "P", hedge_presrcpoint(geo, bhedge));
        bcotan = 1.0 / tan(acos(dot(bptpos - thispos, bptpos - neighpos)));

        areasum += prim(geo, "area", aprim) / 3.0;
        //cotsum += (acotan + bcotan) / 2.0;
        append(weights, (acotan + bcotan) / 2.0);
    }
    append(weights, areasum);
    return weights;
}


#endif
