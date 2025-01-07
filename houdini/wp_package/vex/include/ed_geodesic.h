

#ifndef _ED_GEODESIC_H

#define _ED_GEODESIC_H

#include "ed_defines.h"
#include "ed_poly.h"


#define SS_FLATTENLENGTH ED_FLATTENLENGTH
#define SS_PRESERVELENGTH ED_PRESERVELENGTH

#define POSITION_EPS 0.0001
#define DIRECTION_EPS 0.001



function void projectsurfacespace2(int geo;
    vector origin; vector vec; int startprim;
    int starthedge;
    float preservelength;
    // output references
    int escapehedge;
    int escapeprim;
    vector escapepos;
    vector remainvec;

)
    {
    /* given face and vector to project,
    flatten it into surface space and project it out

    geo MUST be a trimesh

    origin : starting position to project vec

    vec : vector to project through mesh, not normalised

    startprim : face on which projection originates

    preservelength : 0.0 flattens vector totally to tangent space,
    1.0 preserves original length of vector into new direction

    escapeprim : next primitive if vector escapes

    escapehedge: next hedge if vector escapes,
    or -1 if vector terminates in this prim

    escapepos: position on edge at which vector escapes, or termination point

    remainvec: escape vector span "hooked" over the edge of
    the polygon to its neighbour, averaging the faces' normals -
    guarantees vector can be projected directly to next face

    */
    vector null = {0,0,0};
    vector primpos = prim(geo, "P", startprim);
    vector primnormal = prim(geo, "N", startprim);

    // first move on to start prim face
    //origin = lerp(origin, primpos, POSITION_EPS);

    // flatten vector to prim's tangent space
    vector tanvec = projectraytoplane(
        null, normalize(primnormal),
        origin, vec
    );
    tanvec = lerp(tanvec, normalize(tanvec) * length(vec), preservelength);

    vector tandir = normalize(tanvec);

    // does vec terminate on this face?
    escapeprim = -1;
    int endsonface = 0;

    // test skewlines of each edge vector
    vector nearpointonedge, nearpointonvec;

    int testhedge = primhedge(geo, startprim);
    int foundhedge = -1; // check in case no hedges are found
    HedgeRay ray;

    for(int i = 0; i < 3; i++){

        // if starthedge is given, skip it and any equivalents
        if(starthedge != -1){
            if ( (testhedge == starthedge) | (testhedge == hedge_nextequiv(geo, starthedge))){
                testhedge = hedge_next(geo, testhedge);
                continue;
            }
        }

        ray = makehedgeray(geo, testhedge);
        skewlinepoints(
            origin, tandir,
            ray.startpos, ray.dir,
            nearpointonedge, nearpointonvec
        );

        // check for point behind origin - discard if so
        if(dot(tandir, nearpointonedge - origin) < 0){
            testhedge = hedge_next(geo, testhedge);
            continue;
        }

        // check that edge skew point actually lies on edge
        float edgelen = length2(ray.span);
        int onedge = (
            (length2(nearpointonedge - ray.startpos) < edgelen) &&
            (length2(nearpointonedge - ray.endpos) < edgelen)
        );

        // discard if not
        if(!onedge){
            testhedge = hedge_next(geo, testhedge);
            continue;
        }

        // check if vec will stretch to this point - if not, vec terminates
        // on this face
        if(length2(origin - nearpointonedge) > length2(vec)){
            foundhedge = testhedge;
            endsonface = 1;
            break;
        }

        // found the hedge
        foundhedge = testhedge;
        break;
    }

    // if no hedge has been found, an error has occurred
    // move point back to centre of prim
    if(foundhedge == -1){
        escapeprim = startprim;
        escapepos = lerp(origin, primpos, 0.1);
        remainvec = tanvec;
        escapehedge = primhedge(geo, startprim);
        return;

    }

    // if ends on face, complete function
    if(endsonface){
        escapeprim = startprim;
        escapepos = origin + tanvec;
        escapehedge = -1;
        return;
    }

    // transport into next primitive
    escapehedge = hedge_nextequiv(geo, foundhedge);
    escapeprim = hedge_prim(geo, escapehedge);
    vector nextprimpos = prim(geo, "P", escapeprim);
    vector nextprimnormal = prim(geo, "P", escapeprim);
    //escapepos = lerp(nearpointonedge, nextprimpos, 0.0001);
    escapepos = nearpointonedge;

    // shorten vector
    float veclength = max(0, length(vec) - length(origin - escapepos));
    tanvec = tandir * veclength;

    // check if hedge is unshared
    // if so, slide along hedge
    // could also do reflection here for bouncing
    if(hedgeisunshared(geo, foundhedge)){
        remainvec = normalize(ray.dir + (nextprimpos - escapepos) * 0.0001)
         * dot(ray.dir, tanvec);
        return;
    }

    // hook vector over halfway into next prim
    vector edgenormal = normalize(lerp(
        primnormal, nextprimnormal, 0.5
    ));

    // remainvec = reflect(tandir, -edgenormal) * veclength;
    remainvec = reflect(tanvec, -edgenormal);

    return;

}



#endif
