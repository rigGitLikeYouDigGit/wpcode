
#ifndef ED_SUPERCONFORM_H
#define ED_SUPERCONFORM_H
// specific functions for the superconform mesh transfer system

#include "ed_poly.h"

vector gathertensionvec(int ptnum; float power; float masspow){
    /* for each edge, apply its tension by the opposite edge
    (or space between edges) in surface space - this isn't perfect,
    but should overcome the "hump" problem. maybe.
    maybe this shouldn't actually go in the tension term

    0 is active geo, converted to lines
    1 is reference geo, converted to lines
    */
    vector sumdir = {0, 0, 0};
    vector null = {0, 0, 0};
    vector ptnorm = point(0, "N", ptnum);
    vector ptpos = point(0, "P", ptnum);
    float ptmass = point(0, "mass", ptnum);


    int lines[] = pointprims(0, ptnum);
    int neighpts[] = neighbours(0, ptnum);
    int n = len(lines);

    //int primpts[] = primpoints(0,
    vector reflectnorm;
    vector spandir;

    for(int i = 0; i < n; i++){
        float activelength = prim(0, "activelength", lines[i]);
        float restlength = prim(1, "restlength", lines[i]);
        vector neighpos = point(0, "P", neighpts[i]);
        float neighmass = point(0, "mass", neighpts[i]);

        float massdiff = ptmass / neighmass;
        massdiff = 1.0 / massdiff;

        float draw = restlength / activelength / massdiff;
        draw = 1.0 / draw;

        //draw = max(1.0, draw);
        //draw = max(restlength / activelength, 1.0);
        //draw = max(activelength / activelength, 1.0);

        vector span = neighpos - ptpos;
        //float edgemag = max(length(span) - rest, 0);

        // project tension vector flat to normal to avoid crazy powers
        //span = normalize(span) * dot(span, ptnorm);
        //span = normalize(span) * (draw - 1.0);
        span = normalize(span) * (draw);
        sumdir += span;
        }
    sumdir /= len(lines);
    return sumdir;
}


// run over primitives, then interpolate to points
vector gatherprimcompression(
    int primnum
){
    vector primpos = prim(0, "P", primnum);
    // check ratio of area between active face and the reference mesh
    float arearatio = prim(0, "area", primnum) / prim(1, "basearea", primnum);
    vector compressvec = {0, 0, 0};
    foreach( int neighpr; polyneighbours(0, primnum)){
        vector neighdir = normalize(vector(prim(1, "P", neighpr)) - primpos);
        float neigharearatio = prim(0, "area", neighpr) / prim(1, "basearea", neighpr);
        compressvec += -neighdir * ((arearatio / neigharearatio) - 1.0);
    }
    compressvec = compressvec / float(len(polyneighbours(0, primnum))) / 5.0;

    // debug
    int newpt = addgrouppoint(0, primpos, "centre");
    addpointline(0, primpos + compressvec, newpt, "compvec");

    return compressvec;

}

vector gathervertexcompression(
    int vtxnum
){
    // compare distance to centre of prim against reference
    vector compressvec = {0, 0, 0};
    vector vtxpos = vertex(0, "P", vtxnum);

    vector baseprimd = vertex(1, "primcentred", vtxnum);
    vector primd = prim(0, "P", vertexprim(0, vtxnum)) - vtxpos;

    float push = length(baseprimd) / length(primd);
    //push = 1.0 / push;
    float cutoff = 1.1;
    push = max(push, cutoff) - cutoff;
    push = min(push, 2.0);
    //push = pow(push, 2.0);
    compressvec = -normalize(primd) * push;

    setvertexattrib(0, "push", -1, vtxnum, push);


    return compressvec;

}

float getddiff(float dnorm; float dref){
    // stub function for conformality
    return  dref - dnorm;
}

// check conformality of each vertex corner with reference
vector gathervertexconform(
    int vtxnum
){
    vector thispos = vertex(0, "P", vtxnum);
    int vtxhedge = vertexhedge(0, vtxnum);
    int thispt = vertexpoint(0, vtxnum);

    vector primpos = prim(0, "P", vertexprim(0, vtxnum));

    int nextpt = hedge_dstpoint(0, vtxhedge);
    int nextvtx = hedge_dstvertex(0, vtxhedge);
    float nextdot = vertex(0, "confnorm", vtxnum);
    vector nextpos = point(0, "P", nextpt);
    vector nextrefpos = vertex(1, "P", nextvtx);

    int prevpt = hedge_presrcpoint(0, vtxhedge);
    int prevvtx = hedge_presrcvertex(0, vtxhedge);
    float prevdot = vertex(0, "confnorm", vtxnum);
    vector prevpos = point(0, "P", prevpt);
    vector prevrefpos = vertex(1, "P", prevvtx);

    float dnorm = dot(
        normalize(nextpos - thispos),
        normalize(thispos - prevpos)
        );
    setvertexattrib(0, "dnorm", -1, vtxnum, dnorm);

    float refnorm = vertex(1, "confnorm", vtxnum);
    setvertexattrib(0, "dref", -1, vtxnum, refnorm);


    float ddiff = getddiff(dnorm, refnorm);
    setvertexattrib(0, "ddiff", -1, vtxnum, ddiff);


    float refprevnorm = vertex(1, "confnorm", prevvtx);
    float refnextnorm = vertex(1, "confnorm", nextvtx);

    // vector of opposite edge
    vector opposite = prevpos - nextpos;
    vector toprev = prevpos - thispos;
    vector tonext = nextpos - thispos;

    // // clamp vector length
    // float opplength = min(length(opposite), length(prevrefpos - nextrefpos), 1.0);
    // //opplength = length(prevrefpos - nextrefpos);
    // opposite = normalize(opposite) * opplength;
    //
    //
    //
    // // push this point along vector of opposite edge by difference in dot product
    // vector confvec = ((prevdot - refprevnorm) + (refnextnorm - nextdot) ) *  opposite
    //     //+ ((dnorm - refnorm) - refnextnorm) * opposite
    //     ;
    vector confvec =
        (-tonext * getddiff(nextdot, refnextnorm))
        + (-toprev * getddiff(prevdot, refprevnorm)) / 2.0;

    // debug
    vector newptpos = lerp(thispos, primpos, 0.1);
    int newpt = addgrouppoint(0, newptpos, "vtxroot");

    vector debugpos = newptpos + confvec * 0.3;
    //addpointline(0, debugpos, newpt, "debug");


    return confvec;

    // // vector to middle of polygon
    // vector midvec = - (thispos - prim(0, "P", vertexprim(0, vtxnum)));
    //
    // // get magnitude to move this vertex
    // float confmag = (dnorm - refnorm);
    // //confmag /= point(0, "mass", thispt);
    // return (midvec / 2.0) * confmag;
}


#endif
