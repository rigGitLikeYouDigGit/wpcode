
#ifndef _ED_POLY_H

#define _ED_POLY_H
#include "array.h"
#include "ed_vector.h"

// ---- vertices -----

function int[] orderedpointvertices(int geo; int pt){
    // return array of point vertices in order
    int vtxes[];
    int vtx = pointvertex(geo, pt);
    while(vtx > -1){
        append(vtxes, vtx);
        vtx = vertexnext(geo, vtx);
    }
    return vtxes;
}


function int[] orderedpointhedges(int geo; int pt; int starthedge){
    // return array of hedges with this point as source,
    // ordered by continuous prims around this point
    // errors for non-manifold topo
    // points on borders scare me :)
    int checkhedges[];
    int checkhedge = starthedge;
    do{
     append(checkhedges, checkhedge);
     checkhedge = hedge_next( geo, // go backwards in next poly
         // next equiv from hedge with point as source,
         // returns hedge with point as dest
        hedge_nextequiv(geo, checkhedge)
            );
     //checkhedge = pointhedgenext(0, checkhedge);
 } while (checkhedge != starthedge);
    return checkhedges;
}

function int[] orderedpointhedges(int geo; int pt){
    // return array of hedges with this point as source,
    // ordered by continuous prims around this point
    // errors for non-manifold topo
    int starthedge = pointhedge(geo, pt);
    return orderedpointhedges(geo, pt, starthedge);
}

function int[] woundpointvertices(int geo; int startvtx){
     // return array of vertices winding around point
     int starthedge = vertexhedge(geo, startvtx);
     int pt = vertexpoint(geo, startvtx);
     int pthedges[] = orderedpointhedges(geo, pt, starthedge);
     int woundvtxes[];
     foreach(int hedge; pthedges){
         append(woundvtxes, hedge_srcvertex(geo, hedge));
     }
     return woundvtxes;
}


function int vertexnextcont(int geo; int vtx){
    // return the next point vertex or the first
    // if this is a point's last vertex
    int vtxnext = vertexnext(0, vtx);
    if (vtxnext < 0){
        vtxnext = pointvertex(0, vertexpoint(0, vtx));
    }
    return vtxnext;
}

function int woundvertexnext(int geo; int vtx){
    // return the next point vertex or the first
    // if this is a point's last vertex
    return hedge_srcvertex(geo,
        hedge_nextequiv(geo,
            vertexhedge(geo, vtx)));
}

function int woundvertexprev(int geo; int vtx){
    return hedge_dstvertex(geo,
        hedge_nextequiv(geo,
            hedge_prev(geo,
                vertexhedge(geo, vtx))));
}

function int vertexprevcont(int geo; int vtx){
    // return the next point vertex or the first
    // if this is a point's last vertex
    int vtxnext = vertexprev(0, vtx);
    if (vtxnext < 0){
        vtxnext = orderedpointvertices(0, vertexpoint(0, vtx))[-1];
    }
    return vtxnext;
}

function int primvertexnext(int geo; int vtx){
    // return the next vertex in the given primitive,
    return hedge_dstvertex(0, vertexhedge(0, vtx));
}

function int primvertexprev(int geo; int vtx){
    // return the next vertex in the given primitive,
    return hedge_presrcvertex(0, vertexhedge(0, vtx));
}

function vector[] vertexvectors(int geo; int cornervtx){
    // get vectors from given vertex in primitive to neighbours

    vector result[];
    resize(result, 2);

    int nextvtx = primvertexnext(geo, cornervtx);
    int prevvtx = primvertexprev(geo, cornervtx);

    result[0] = vector(vertex(geo, "P", cornervtx)) -
        vector(vertex(geo, "P", prevvtx));
    result[1] = vector(vertex(geo, "P", cornervtx)) -
        vector(vertex(geo, "P", nextvtx));
    return result;
}

function vector[] vertexvectorsnorm(int geo; int cornervtx){
    // as above, but normalised
    vector vecs[] = vertexvectors(geo, cornervtx);
    vector result[] = array(
        normalize(vecs[0]),
        normalize(vecs[1])
    );
    return result;
}





// ---- points -----
function int addgrouppoint( int geo; vector pos; string name){
    int pt = addpoint(geo, pos);
    setpointgroup(geo, name, pt, 1);
    return pt;
}

function int iscornerpoint( int geo; int pointId ){
    // is this point on the corner of a quad mesh?
    return ( neighbourcount(geo, pointId) == 2);
    }

function float sumpointdistances( int geo; int points[]){
    // sums the distance between points, in order of array
    float sum = 0.0;
    for( int i = 1; i < len(points); i++){
        vector a = point(geo, "P", points[i - 1]);
        vector b = point(geo, "P", points[i]);
        sum += distance( a, b );
    }
    return sum;
}

function int initpointattrs(int geo){
    // just set some useful attributes on to the points
    // to be run in detail for now
    addattrib(0, "point", "relpos", vector(set(1, 1, 1)));
    for (size_t i = 0; i < npoints(0); i++) {
        setpointattrib(geo, "id", i, i);
        if (i < 1){
            vector relpos = set(0, 0, 0);
            setpointattrib(geo, "relpos", i, relpos);
            continue;
        }
        vector relpos = point(geo, "P", i)
            - point(geo, "P", i-1);
        setpointattrib(geo, "relpos", i, relpos);
    }
    return 1;
}

function vector[] pointpositions(int geo;){
    // return point positions as vector array
    vector result[];
    for(int i=0; i < npoints(geo); i++){
        append(result, vector(point(geo, "P", i)));
    }
    return result;
}

function int setpointpositionsfromarray(int geo; vector poses[]){
    // given array of vectors, set all point positions
    for( int i = 0; i < len(poses); i++){
        setpointattrib(geo, "P", poses[i]);
    }
    return 1;
}

// ---- lines ----
function int[] addpointline(int geo; vector pos; int parentpt){
    // addpoint, but automatically adds a polyline from parent
    int npt = addpoint(0, pos);
    int nprim = addprim(0, "polyline", parentpt, npt);
    int result[] = array(npt, nprim);
    return result;
}

function int[] addpointline(int geo; vector pos; int parentpt; string group){
    // addpoint, but automatically adds a polyline from parent
    int pts[] = addpointline(geo, pos, parentpt);
    setpointgroup(geo, group, pts[0], 1);
    setprimgroup(geo, group, pts[1], 1);
    return pts;
}



int connectpointsbyattr( int geo; int ptnum; float range; string attr){
    vector pos = point(geo, "P", ptnum);
    int neighbours[] = nearpoints(geo, pos, range, 30 );
        foreach( int pt; neighbours){
            //if( pt > i@ptnum ){
            // if( pt > point(geo, "ptnum", ptnum) ){
            if( pt > ptnum ){
                int id = point(0, "id", pt);
                if( id == id){
                    int line = addprim(0, "polyline", ptnum, pt);
                    return line;
                    //break;
                    }
                }
            }
    return -1;
}

function int[] insertpoint(int outerprim; int pt){
    // given one primitive and one point,
    // insert point in primitive and triangulate
    // for each half
    int hedge = primhedge(0, outerprim);
    //int pts[] = primpoints(0, outerprim);
    int newprims[];

    for (size_t i = 0; i < len(primpoints(0, outerprim)); i++)
    {
        int pta = hedge_srcpoint(0, hedge);
        int ptb = hedge_dstpoint(0, hedge);
        int newprim = addprim(0, "poly",  pta, ptb, pt);
        append(newprims, newprim);
        hedge = hedge_next(0, hedge);
    }
    removeprim(0, outerprim, 1);

    return newprims;
}



function int[] insertpoints(string weldpts; string collisionprims;
    string newgrp; int mode){
    // inserts new points into collision prim faces,
    // corresponding to weld pts
    // mode - 0 : nearest point, 1 : normal project

    // this can be partially parallelised - can only insert in one base face at a time

    int pts[] = expandpointgroup(1, weldpts);
    int addedpts[];
    foreach(int pt; pts){
        // nearest point
        vector pos = point(1, "P", pt);
        vector uvw;
        vector hitpos;
        int hitprim;
        if (mode == 0){
            xyzdist(0, collisionprims, pos, hitprim, uvw);
        }
        if (mode == 1){
            vector dir = -point(1, "N", pt);
            intersect(0, collisionprims, pos, dir, hitpos, uvw);
            xyzdist(0, collisionprims, hitpos, hitprim, uvw);
            // intersect(0, pos, dir, hitpos, uvw);
            // xyzdist(0, hitpos, hitprim, uvw);
        }
        hitpos = primuv(0, "P", hitprim, uvw);
        int newpt = addpoint(0, hitpos);

        int newprims[] = insertpoint(hitprim, newpt);
        //removeprim(0, hitprim, 0);
        foreach(int newprim; newprims){
            setprimgroup(0, collisionprims, newprim, 1);
        }
        append(addedpts, newpt);
    }
    return addedpts;
}

function int insertpolylinevertex(int primnum; vector pos ){
    // add given position as vertex - will not reorder vertices
    // inserts in segment holding nearest point

    int hitprim;
    vector nearuvw;

    // get uv on desired polyline
    xyzdist(0, primnum, pos, hitprim, nearuvw);

    // get previous and next vertices to wire in
    int prevvtx = nearuvw.x * primvertexcount(0, primnum);
    int nextvtx = nearuvw.x * primvertexcount(0, primnum) + 1;

    // add point
    int npt = addpoint(0, pos);

    // add as vertex
    int newvtx = addvertex(0, primnum, npt);

    vector hitpos, uvw;


}




// ---- half-edges ------
#define prevloophedge(geo, hedge)\
    (hedge_prev(geo, hedge_nextequiv(geo, hedge_prev(geo, hedge))))

// #define nextloophedge(geo, hedge)\
//     (hedge_next(geo, hedge_nextequiv(geo, hedge_next(geo, hedge))))

function int nextloophedge(int geo, hedge){
    return hedge_next(geo, hedge_nextequiv(geo, hedge_next(geo, hedge)));
}

function int[] outgoinghedges(int geo, pt){
    /* return ordered list of all hedges with pt as source.
    on border, at least one hedge will only be incoming,
    so will not be returned here

    first go forwards until we find unshared edge or repeat - catches case where
    pointhedge() returns an edge in middle of fan
    */
    int starthedge = pointhedge(geo, pt);
    int currenthedge = int(starthedge);
    int testhedges[];
    while(
        (hedge_equivcount(geo, currenthedge) > 1) &&
        (hedge_next(geo, hedge_nextequiv(geo, currenthedge)) != starthedge)
    ){
        append(testhedges, currenthedge);
        currenthedge = hedge_next(geo, hedge_nextequiv(geo, currenthedge));
    }

    // now go backwards
    starthedge = currenthedge;
    int hedges[];
    do{
        append(hedges, currenthedge);
        currenthedge = hedge_nextequiv(geo, hedge_prev(geo, currenthedge));
    }
        while(
        (hedge_equivcount(geo, hedge_nextequiv(geo, hedge_prev(geo, currenthedge))) > 1) &&
        (hedge_nextequiv(geo, hedge_prev(geo, currenthedge)) != starthedge)
    );
    return hedges;
}

function int[] adjacenthedgesset(int geo, pt){
    /* return an unsorted list of ajacent hedges */
    int hedges[];
    foreach(int npt; neighbours(geo, pt)){
        append(hedges, pointedge(geo, pt, npt));
    }
    return hedges;
}

function int[] adjacenthedges(int geo, pt){
    /* return ordered list of all hedges touching pt

    first go forwards until we find unshared edge or repeat - catches case where
    pointhedge() returns an edge in middle of fan
    */
    int starthedge = pointhedge(geo, pt);
    int currenthedge = int(starthedge);
    int testhedges[];
    append(testhedges, currenthedge);

    while(
        (hedge_equivcount(geo, currenthedge) > 1) &&
        (hedge_next(geo, hedge_nextequiv(geo, currenthedge)) != starthedge)
    ){
        currenthedge = hedge_next(geo, hedge_nextequiv(geo, currenthedge));
        append(testhedges, currenthedge);

    }

    // now go backwards
    starthedge = currenthedge;
    int hedges[];
    do{
        append(hedges, currenthedge);
        append(hedges, hedge_prev(geo, currenthedge));
        currenthedge = hedge_nextequiv(geo, hedge_prev(geo, currenthedge));

    }
        while(
        (hedge_equivcount(geo, hedge_nextequiv(geo, hedge_prev(geo, currenthedge))) > 1) &&
        (hedge_nextequiv(geo, hedge_prev(geo, currenthedge)) != starthedge)
    );
    if(hedge_equivcount(geo, starthedge) > 1){
        append(hedges, currenthedge);
        append(hedges, hedge_prev(geo, currenthedge));
    }

    return hedges;
}



struct HedgeRay{
    int index; // hedge index
    int prim; // prim index
    int startpt; // start point index
    int endpt; // end point index
    int startvtx; // start vertex index
    int endvtx; // end vertex index
    vector startpos; // start position
    vector endpos; // end position
    vector span;
    vector dir; // normalised direction
};

function HedgeRay makehedgeray(int geo, hedge){
    vector startpos = point(geo, "P", hedge_srcpoint(geo, hedge));
    vector endpos = point(geo, "P", hedge_dstpoint(geo, hedge));
    vector span = endpos - startpos;
    vector dir = normalize(span);
    return HedgeRay(
        hedge,
        hedge_prim(geo, hedge),
        hedge_srcpoint(geo, hedge),
        hedge_dstpoint(geo, hedge),
        hedge_srcvertex(geo, hedge),
        hedge_dstvertex(geo, hedge),
        startpos,
        endpos,
        span,
        dir
    );

}

// consider
struct HalfEdge {
    int equivalents[];
    int prim;
};

/* is this of any use at all? or are the basic functions enough?
anything to get more high-level control, but I don't think it
plays well with the rest of the ethos we have here
*/

function int[] hedgepoints( int geo; int hedge ){
    // returns points belonging to half edge
    int out[] = array( hedge_srcpoint(geo, hedge),
        hedge_dstpoint(geo, hedge) );
    return out;
}

function int hedgepointopposite( int geo; int hedge; int pt ){
    // return the opposite point in hedge
    int pts[] = hedgepoints(geo, hedge);
    removevalue(pts, pt);
    return pts[0];
}

function int[] halfedgeequivalents( int geo; int hedge ){
    // returns the other half edges belonging to same main edge
    int edges[];
    int n = hedge;
    do{
        append(edges, n);
        n = hedge_nextequiv(geo, n); // WARN
    }while(n != hedge);
    return edges;
};

function int hedgeisunshared( int geo; int hedge ){
    // returns 1 if hedge is unshared else 0
    return ( hedge == hedge_nextequiv(geo, hedge));
}

function int[] allhedgeequivalents( int geo; int hedge ){
    // returns input and all equivalents
    int edges[] = halfedgeequivalents( geo, hedge);
    //append( edges, hedge );
    return edges;
};

function int[] primhalfedges( int geo; int prim ){
    // return all halfedges in primitive
    int edges[];
    int current = primhedge( geo, prim );
    int start = current;
    do{
        append(edges, current);
        current = hedge_next( geo, current );

    }while(start != current);
    return edges;
};

function int[] primhalfedgesexcept( int geo; int prim; int except){
    // returns all primitive half edges except hedge specified
    // ALSO REMOVE ALL HEDGES EQUIVALENT TO EXCEPT
    int edges[] = primhalfedges( geo, prim );
    foreach( int remove; allhedgeequivalents( geo, except )){
        removevalue( edges, remove);
    }
    return edges;
}

function int[] hedgeprims( int geo; int hedge ){
    // returns all primitives containing this half edge or equivalents
    int out[];
    foreach( int h; allhedgeequivalents(geo, hedge)){
        append( out, hedge_prim(geo, h));
    }
    return out;
}

function int primpointshedge( int geo; int prim; int pointa, pointb){
    // return half edge on prim between given points
    // return intersect(primhalfedges(geo, prim), allhedgeequivalents(
    //     geo, pointedge(geo, pointa, pointb)))[0];
    int hedge = primhedge(geo, prim);
    int basehedge = hedge;
    // for (int i = 0; i < primvertexcount(geo, prim); i++) {
    do {
        if(
            ((hedge_srcpoint(geo, hedge) == pointa) &&
            (hedge_dstpoint(geo, hedge) == pointb))
            ||
            ((hedge_srcpoint(geo, hedge) == pointb) &&
            (hedge_dstpoint(geo, hedge) == pointa))
        ){
            return hedge;
        }
        hedge = hedge_next(geo, hedge);
    }while(basehedge!=hedge);
    return -1;
}

function vector[] hedgeray( int geo; int hedge ){
    // return pos and dir corresponding to source and dest hedge point
    int pts[] = hedgepoints(geo, hedge);
    vector pos = point(geo, "P", pts[0]);
    return array(pos, point(geo, "P", pts[1]) - pos );
}


// higher topo functions

// time for a diagram

/*

starthedge              targethedge
    \                       \
     \                       \
================== O ===================
    ---->        | I        ---->
                 | I ^
        primA    v I |       primB
                   I |
                   I
*/

function int pointissingularity( int geo; int pt){
    /* point may be singularity for 2 reasons:
     - topological pole (3, 5 or more adjacent polygons)
     */

     int onborder = 0;
     // check if any neighbouring hedges are borders
     foreach(int hedge; orderedpointhedges(geo, pt)){
         if(hedge_nextequiv(geo, hedge) == hedge){
             onborder = 1;
             break;
         }
     }
     // check if point is singularity based on if it's on border
     if (onborder){
         return (len(pointprims(geo, pt)) != 2);
     }
     return neighbourcount(geo, pt) != 4;

}

function int edgeloopshouldterminate(int geo; int testhedge){
    /* direction of hedge is considered.
     - if dest point is a singularity
     - if edge is perpendicular to an unshared border
     return 1
     */
     if(pointissingularity(geo, hedge_dstpoint(geo, testhedge))){
         return 1;
     }
     // check that hedge does not already lie on a border
     if(hedge_equivcount(geo, testhedge) == 2){
         // check if either diverging hedges are unshared
         if(
             (hedge_equivcount(geo, hedge_next(geo, testhedge)) == 1) +
             (hedge_equivcount(geo,
                 hedge_prev(geo,hedge_nextequiv(geo, testhedge))) == 1)){
                     return 1;
                 }

     }
     return 0;

}


function int[] edgeloop( int geo; int seedhedge ){
    int outhedges[];
    int currenthedge = seedhedge;
    do{
        append(outhedges, currenthedge);
        currenthedge = nextloophedge(geo, currenthedge);
    }while(
        (currenthedge != seedhedge) &&
        (neighbourcount(geo, hedge_srcpoint(geo, currenthedge)) == 4)
    );
    // backwards
    currenthedge = prevloophedge(geo, seedhedge);
    if (find(outhedges, currenthedge) > -1){
        return outhedges;
    }

    while(
        (currenthedge != seedhedge) &&
        (neighbourcount(geo, hedge_dstpoint(geo, currenthedge)) == 4)
        ){
        append(outhedges, currenthedge);
        currenthedge = prevloophedge(geo, currenthedge);
    }
    return outhedges;
}

function int[] pointsfromhedges( int geo; int hedges[] ){
    // return all points included in halfedge selection
    int out[];
    foreach( int hedge; hedges){
        // out = union(out, hedgepoints(geo, hedge));
        int temppts[] = hedgepoints(geo, hedge);
        //out = union(temppts, tempout);
        out = union(out, temppts);
    }
    return sort(out);
}

function int[] pointloopfrompoints( int geo; int a; int b){
    // returns all points lying on edge loop of a and b
    int columnhedge = pointedge( geo, a, b );
    int columnhedges[] = edgeloop( geo, columnhedge );
    int columnpts[] = pointsfromhedges( geo, columnhedges );
    return columnpts;
}

function int maprowscolumns( int geo; int corner; int columndir ){
    // sets int attributes denoting row and column of each point
    // assumes quad mesh
    // columndir is next point in column, must be neighbour
    if( !iscornerpoint( geo, corner )){
        return -1; // do it right
    }

    // get initial column points
    int columnpts[] = pointloopfrompoints( geo, corner, columndir );
    int rowpts[];
    // iterate columns
    foreach( int i; int columnpt; columnpts){
        // get transverse points
        int transverse[] = subtract( neighbours( geo, columnpt), columnpts );
        rowpts = pointloopfrompoints( geo, columnpt, transverse[0] );

        // iterate rows
        foreach( int n; int rowpt; rowpts){
            setpointattrib( geo, "column", rowpt, i);
            setpointattrib( geo, "row", rowpt, n);
        }
    }
    setdetailattrib( geo, "ncolumns", len(columnpts));
    setdetailattrib( geo, "nrows", len(rowpts));
    return 1;
}


function vector halfedgemidpoint( int geo; int hedge ){
    vector startPos = point( geo, "P", hedge_srcpoint( geo, hedge ) );
    vector endPos = point( geo, "P", hedge_dstpoint( geo, hedge ) );
    return (endPos + startPos) / 2.0 ;
};


function int[] crawlmesh5(int geo;
     int basehedge;
     int reversedir;
     int baseindex;
     int foundpts[];
     int foundprims[];
     int twinpts[];
    ){
    int newhedges[];

    // early out if prim has been processed
    //int primfound = prim(0, "found", vertexprim(0, basevtx));
    int baseprim = hedge_prim(0, basehedge);
    int primfound = foundprims[baseprim];
    if (primfound > 0){
        return newhedges;
    }
    //return newvtxes;
    int localiter = 0;
    int currenthedge = basehedge;
    int vtxpt, lookuptwin;
    do{
        if(localiter > 10){
            printf("failsafe-break_");
            break;
        }

        append(newhedges, hedge_nextequiv(geo, currenthedge));

        vtxpt = hedge_dstpoint(0, currenthedge);
        lookuptwin = foundpts[vtxpt];
        if(lookuptwin == -1){
            setpointattrib(0, "twin", vtxpt, baseindex);
            foundpts[vtxpt] = baseindex;
            twinpts[baseindex] = vtxpt;
            baseindex++;
        }
        append(newhedges, hedge_nextequiv(0, currenthedge));
        if (reversedir){
            currenthedge = hedge_prev(0, currenthedge);
        }
        else{
            currenthedge = hedge_next(0, currenthedge);
        }
        localiter++;
    }while (currenthedge != basehedge);

    foundprims[baseprim] = 1;

    setprimattrib(0, "found", hedge_prim(0, basehedge), 1);

    return newhedges;
}



// Higher functions still combining topo with spatial info

function matrix3 tangentmatrix(int geo;
    int face){
        // return orientation matrix for polygon face
        // oriented to face gradient
        string geos = opfullpath(".");
        vector2 samplepos = set(0.5, 0.5);
        vector grad = normalize(primduv(geos, face, samplepos, 1, 1));
        vector N = normalize(prim(geo, "N", face));
        vector bitan = cross(grad, N);

        return matrix3(set(grad, N, bitan));
    }


#define SS_FLATTENLENGTH ED_FLATTENLENGTH
#define SS_PRESERVELENGTH ED_PRESERVELENGTH

function void projectsurfacespace(int geo;
    vector origin; vector dir; int face; int starthedge; int ptnum;
    int lengthmode;
    float edgehingeweight;
    //int hookoutput;
    // output references
    int escapehedge;
    vector escapepos;
    vector outputlin;
    vector outputhook;

    int debug; // temp
    int success;
    int debugparent; // parent to connect debug lines
)
    {
    /* given face and vector to project,
    flatten it into surface space and project it out

    lengthmode: either squash a high-angle vector
    to its direct projection,
    or restore its length in surface space

    edgehingeweight: for hook projection, weight the hinge axis
    to either the escape edge vector, or cross product between
    primitive normals

    escapehedge: returns the prim hedge index crossed by the
    flattened vector, or -1 if vector terminates in this prim
    escapepos: position on edge at which vector escapes

    outputlin : escape vector linear span beyond given face,
    or within if it terminates

    outputhook: escape vector span "hooked" over the edge of
    the polygon to its neighbour, averaging the faces' normals -
    guarantees vector can be projected directly to next face

    if ray falls within primitive, escapehedge will be -1

    we assume that origin will be located within the projective face -
    if not ray will immediately intersect with a halfedge and return

    */

    if(length(dir) < EPS){
        //return;
    }

    vector up = {0, 1, 0};
    vector null = {0,0,0};

    vector midpos = prim(geo, "P", face);

    // move origin a tiny amount into this prim
    origin = lerp(origin, midpos, EPS * 10.0);

    int midpointpt;
    int originpt;

    if(debug){
        int midpointpt = addgrouppoint(0, midpos, "test");
        int originpt = addgrouppoint(0, origin, "origin");
    }

    if(starthedge > -1){
        vector starthray[] = hedgeray(geo, starthedge);
        if(debug){
            addpointline(0, lerp(midpos,
                    starthray[0] + starthray[1] / 2.0,
                    0.9),
                originpt, "starthedge");
            }
        }

    // matrix to rotate all points to surface space
    matrix3 orientmat = tangentmatrix(geo, face);
    float dirlength = length(dir);

    vector normal = set(
        getcomp(orientmat, 1, 0),
        getcomp(orientmat, 1, 1),
        getcomp(orientmat, 1, 0));

    normal = normalize(prim(geo, "N", face));

    setpointattrib(0, "N", ptnum, normal);

    vector tanvec = projectraytoplane(
        null, normalize(normal),
        origin, dir,
        lengthmode
    );
    if(debug){  addpointline(0, origin + tanvec, originpt, "dirsubtan");}
    vector tandir = normalize(tanvec);

    // iterate over halfedges
    //int start = hedge_next(geo, starthedge);
    int start = starthedge;
    if(starthedge == -1){
        start = primhedge( geo, face);
    }
    int current = start;

    int hedgescovered[];
    vector currentray[];

    vector endptvec = dir - midpos;


    float vecdots[];
    vector vtxvecs[];
    int sortpts[];
    vector vtxpos, vtxvec;
    int currentpt;
    int nextpt;
    int foundhedge = -1;

    // check barycentric coords for each pair
    // of edge vertices
    // shrink down endpt vec for now, the
    vector ptposa;vector ptposb;
    float testbcoords[];

    vector testskewa;
    vector testskewb;
    float skewdistmins[];
    vector testhedgeray[];

    //addpointline(0, origin + tanvec, originpt, "tanv");

    int skewhedge = -1;
    float skewdistmin = 10000000000.0;
    float testskewdist;
    vector skewposmin;
    vector foundtanvec;
    int skewminpts[];
    vector skewminposes[];

    do{
        currentpt = hedge_srcpoint(geo, current);
        ptposa = point(geo, "P", currentpt);
        nextpt = hedge_dstpoint(geo, current);
        ptposb = point(geo, "P", nextpt);

        testhedgeray = hedgeray(geo, current);
        vector midtestpos = testhedgeray[0] + testhedgeray[1] / 2.0;

        skewlinepoints(
            origin, tandir,
            testhedgeray[0], normalize(testhedgeray[1]),
            testskewa, testskewb
        );
        // check skewpoint not behind direction of vector
        if(dot(normalize(testskewb - origin), tandir) <= 0.0){
            if(debug){
                addpointline(0, midtestpos, midpointpt, "hedgebehind");
            }
            current = hedge_next(geo, current);
            continue;
        }

        if(debug){int newskewpt = addpointline(0, testskewb, midpointpt, "skewptb")[0];}

        // check distance to skew point
        testskewdist = length2(testskewb - origin);

        if(testskewdist < skewdistmin){
            foundhedge = current;
            skewdistmin = testskewdist;
            skewposmin = testskewb;
            skewminpts = array(currentpt, nextpt);
            skewminposes = array(ptposa, ptposb);
            foundtanvec = tanvec;
        }

        current = hedge_next(geo, current);

    }while (start != current);
    if(foundhedge == -1){
        setpointgroup(0, "error", ptnum, 1);
        return;
        error("missing hedge");
    }

    tanvec = foundtanvec;

    tandir = normalize(tanvec);

    vector hray[] = hedgeray(geo, foundhedge);
    vector hmidpos = hray[0] + hray[1] / 2.0;

    if(debug){
        addpointline(0, origin + dir, originpt, "dir");
        addpointline(0, origin + tanvec, originpt, "dirt");
        addpointline(0, hmidpos, midpointpt, "foundmid");
    }


    // check if vector terminates on face
    float tanskewdot = dot(tanvec, skewposmin - origin);
    int onface = (
        (tanskewdot > dot(tanvec, tanvec) )
    );
    // printf("%f %f ", tanskewdot, dot(tanvec, tanvec));
    // printf(itoa(onface) );
    if(onface){
        // terminates on face
        escapehedge = -1;
        outputlin = tanvec * 0.99;
        outputlin = projectpostoplane(
            null, normalize(normal), outputlin); // correct in general
        if(debug){addpointline(0, origin + outputlin, originpt, "outlin");}

        setpointgroup(0, "inface", ptnum, 1);

        outputhook = outputlin;
        success = 1;
        return;
    }

    // get intersection of hedge and vector with skewlines
    int pta = skewminpts[0];
    int ptb = skewminpts[1];
    vector ptapos = skewminposes[0];
    vector ptbpos = skewminposes[1];

    //int crosshedge = primpointshedge(geo, face, pta, ptb);
    int crosshedge = foundhedge;
    //vector hray[] = hedgeray(geo, crosshedge);
    vector hedgedir = normalize(hray[1]);

    escapehedge = crosshedge;
    //escapepos = skewposmin;
    int nextprim = hedge_prim(geo, hedge_nextequiv(geo, escapehedge));
    vector nextnormal = normalize(prim(geo, "N", nextprim));
    vector nextpos = prim(geo, "P", nextprim);



    escapepos = lerp(skewposmin, nextpos, EPS*10.0);
    float linlength = max(
        ((length(tanvec) - length(escapepos - origin))),
        0.0);
    //printf("%f ", linlength );
    //addpointline(0, escapepos + tandir * 0.5, escpt, "outlinbase");
    outputlin = tandir * linlength;

    // check if hedge is unshared
    // slide along hedge
    if(hedgeisunshared(geo, foundhedge)){
        //escapepos =  lerp(skewposmin, midpos, EPS);
        outputlin = normalize(hray[1]) * dot(normalize(hray[1]), tanvec);
        outputhook = outputlin;
        //escapehedge = -1;
        success = 1;
        return;

    }


    // hook vector over to next face if it exists

    // reflect from cross of hedge dir with normal
    vector edgenormal = normalize(lerp(
        normalize(normalfromtripoints(midpos, hray[0], hray[0] + hray[1])),
        -normalize(normalfromtripoints(nextpos, hray[0], hray[0] + hray[1])),
        0.5
    ));
    if(dot(edgenormal, normal) < 0.0){
        edgenormal = -edgenormal;
    }
    outputhook = reflect(normalize(outputlin), -edgenormal) * length(outputlin);

    //outputhook = outputlin;
    if(debug){
        int escpt = addgrouppoint(0, escapepos, "escape");
        addpointline(0, escapepos + outputlin, escpt, "outlin");
        addpointline(0, escapepos + edgenormal, escpt, "edgenormal");
        addpointline(0, escapepos + outputhook, escpt, "outhook");
    }


    success = 1;

    return;

}

function int[] primsbetweenpoints(int geo;
    int pta; int ptb;
    int activeprim; int visitedprims[];
    int success;
){
    // breadth search from a to b,
    // adding primitives to visited
    int activepts[] = primpoints(geo, activeprim);
    if(IN(activepts, ptb)){
        success = 1;

    }
    return activepts;
}


function int[] oppositepointhedge(int geo, pt, hedge){
    // return the hedge going into pt, opposite to given hedge
    // if odd, returns 2 of them
    int result[];
    // ensure pt is dest point of hedge
    if(pt == hedge_srcpoint(geo, hedge)){
        hedge = hedge_nextequiv(geo, hedge);
    }

    // odd or even
    int npts = neighbourcount(geo, pt);
    int n = npts / 2 + 1;
    int foundhedge = hedge;
    for (int i = 0; i < n; i++) {
        foundhedge = hedge_next(geo,
            hedge_nextequiv(geo,
            hedge_next(geo, foundhedge)));
        if((n - i) <= 2){
            append(result, foundhedge);
        }
    }
    return result;
}

// function vector[] vertexvectors(int geo, vtx){
//     vector ptpos = point(geo, "P", vertexpoint(geo, vtx));
//     vector result[] = array(
//         point(geo, "P", hedge_dstpoint(geo, vertexhedge(geo, vtx))) - ptpos,
//         point(geo, "P", hedge_presrcpoint(geo, vertexhedge(geo, vtx))) - ptpos
//     );
//     return result;
// }



// interpolates an attribute barycentrically on given face
#define BARY_INTERP_FN(type)\
function type baryinterppointattr(int geo; int triface; \
    vector pos; string atname) \
    { \
    int primpts[] = primpoints(geo, triface); \
    float coords[] = barycoords(primpts[0], primpts[1], primpts[2], pos);\
    type result = ( \
        point(geo, atname, primpts[0]) * coords[0] + \
        point(geo, atname, primpts[1]) * coords[1] + \
        point(geo, atname, primpts[2]) * coords[2] \
    ); \
    return result; \
}

BARY_INTERP_FN(float)
BARY_INTERP_FN(vector)

// interpolates an attribute barycentrically on given face to get gradient
function vector barygradientpointattr(int geo; int triface;
    vector pos; string atname)
    {
    int primpts[] = primpoints(geo, triface);
    vector ptpositions[] = array(
        point(geo, "P", primpts[0]),
        point(geo, "P", primpts[1]),
        point(geo, "P", primpts[2])
    );
    float coords[] = barycoords(
        ptpositions[0],
        ptpositions[1],
        ptpositions[2], pos);

    vector result = (
        point(geo, atname, primpts[0]) *
        (ptpositions[0] - pos) * coords[0] +

        point(geo, atname, primpts[1]) *
        (ptpositions[1] - pos) * coords[1] +

        point(geo, atname, primpts[2]) *
        (ptpositions[2] - pos) * coords[2]
    ) / 3.0;
    return result;
}


function int[] inserttrivertex(int geo; int tri; int pt){
    /* add a vertex into the given triangle, while replacing
    the primitives seamlessly

    returns [ new vertex id, new primitive id ]
    */
    int result[];
    resize(result, 2);
    int nvtx = addvertex(geo, tri, pt);
    // restore tri primitive to new gap
    //int npta = vertexnext(0, nvtx);
    //int nptb = vertexprev(0, nvtx);
    //int npts[] = neighbours(0, pt);


    //int npr = addprim(0, "poly", pt, npts[1], npts[0]);
    //int npr = addprim(0, "poly", pt, vertexpoint(0, nptb), vertexpoint(0, npta));
    result[0] = nvtx;
    //result[1] = npr;
    return result;

}





#endif
