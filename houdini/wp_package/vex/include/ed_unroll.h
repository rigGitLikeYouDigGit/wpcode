#ifndef _ED_UNROLL_H

#define _ED_UNROLL_H

#include "array.h"
#include "ed_poly.h"

/*
for unfolding or unrolling polygons
*/


function int gatherfoldangles(int geo; string normalattr){
    // gather attributes to add to points
    int ids[];
    float spans[];
    float angles[];
    matrix3 frames[];
    float totalspans[];
    float totalspan = 0.0;
    // initpointattrs(geo);
    //addattrib(0, "point", "relpos", vector(set(1, 1, 1)));


    for (size_t i = 0; i < npoints(geo); i++) {
        append(ids, i);
        vector n = point(geo, normalattr, i);
        matrix3 basetf = point(geo, "transform", i);
        if (i == 0){
            append(spans, 0);
            append(totalspans, 0);
            append(angles, 0);
            append(frames, basetf);
            continue;
        }  //

        // get distance to previous point
        vector pos = point(geo, "P", i);
        vector ppos = point(geo, "P", i - 1);
        vector rpos = pos - ppos;

        setpointattrib(0, "relpos", i, rpos);

        float span = length(rpos);
        append(spans, span);
        totalspan = totalspan + span;
        setpointattrib(0, "span", i, span);
        setpointattrib(0, "totalspan", i, totalspan);

        append(totalspans, totalspan);

        int id2 = point(0, "id", i);
        setpointattrib(0, "id2", i, id2);

        if (i == 1){ // can't fold first segment
            append(angles, 0);
            setpointattrib(0, "frame", i, basetf);
            append(frames, basetf);
            continue;
        }

        // vector of previous edge
        vector pppos = point(geo, "P", i - 2);
        vector prevvec = normalize(ppos - pppos);

        // get transform frame for previous edge
        //matrix3 prevframe = maketransform(prevvec, n);
        matrix3 prevframe = frames[i - 1];

        // make transform for this edge
        matrix3 thisframe = maketransform(normalize(rpos), n);
        thisframe = basetf;

        //addpointline(0, ppos + thisframe * set(0, 0, span), i-1, "framepos");

        // relative transform between the two
        matrix3 relframe = invert(prevframe) * thisframe ;
        //relframe = thisframe;

        append(frames, thisframe);
        setpointattrib(0, "frame", i, relframe);


    }
    //vector poses[] = pointpositions(geo);
    //float length = sumpointdistances(geo, ids);

    return 1;

}


function int rollpoints(int geo; float weights[]){
    // unroll each point sequentially according to weight
    vector baseposes[] = pointpositions(geo);
    vector poses[] = pointpositions(geo);

    matrix3 prevframe = point(0, "frame", 1);

    for (size_t i = 0; i < npoints(0); i++) {
        if( i < 2){ // not first segment
            continue;
        }
        float weight = weights[i];


        // reconstruct outer product frame

        vector pos = poses[i];
        vector ppos = poses[i - 1];
        vector pppos = poses[ i - 2];
        vector oldrpos = point(0, "relpos", i);
        vector newrpos = pos - ppos;

        vector prevvec = ppos - pppos;
        vector n = normalize(point(geo, "N", i - 1));
        float span = point(geo, "span", i);
        vector flatpos = (normalize(prevvec)* span );

        matrix3 id = ident();

        // retrieve transform for this edge
        matrix3 relframe = point(0, "frame", i);
        relframe = slerp(id, relframe, weight);
        vector tfpos = prevframe * relframe * set(0, 0, span);

        vector targetpos = ppos + tfpos;
        prevframe = prevframe * relframe;

        vector newpos = targetpos;

        vector resultpos = newpos;
        setpointattrib(geo, "P", i, resultpos);
        poses[i] = resultpos;

    }
    setdetailattrib(geo, "weights", weights);
    return 1;
}

function int rollpoints(int geo; float weight){
    float weights[] = initarray(npoints(geo), weight);
    return rollpoints(geo, weights);
}

function int rollpoints(int geo; string weightattr){
    float weights[] = initarray(npoints(geo), 0.0);
    for (size_t i = 0; i < npoints(geo); i++) {
        float weight = point(geo, weightattr, i);
        weights[i] = weight;
    }
    return rollpoints(geo, weights);
}

#endif
