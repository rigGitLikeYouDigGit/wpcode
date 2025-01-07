
#ifndef _ED_VECTOR_H

#define _ED_VECTOR_H
#include "array.h"

#define EPS 0.000001

// lib file for pure vector operations
#define ED_FLATTENLENGTH 0
#define ED_PRESERVELENGTH 1

function vector projectpostoplane(
    vector normalpos;
    vector normaldir;
    vector projectpos;
){
    // project given position to plane formed by normalpos and normaldir
    return (dot(normaldir, normalpos - projectpos) * normaldir) + projectpos;
}

function vector projectraytoplane(
    vector normalpos;
    vector normaldir;
    vector raypos;
    vector rayspan;
    int lengthmode
){
    // project given ray to surface plane, optionally preserving length
    // assumes ray pos is already on plane
    vector projspan = (dot(normaldir, normalpos - rayspan) * normaldir) + rayspan;
    if(lengthmode == ED_PRESERVELENGTH){
        vector raydir = normalize(projspan);
        projspan = raydir * length(rayspan);
    }
    return projspan;
}

function vector projectraytoplane(
    vector normalpos;
    vector normaldir;
    vector raypos;
    vector rayspan;
){
    // project given rayspan to plane
    // assumes ray pos is already on plane
    vector projspan = (dot(normaldir, normalpos - rayspan) * normaldir) + rayspan;
    return projspan;
}


function vector ed_reflect(vector dir; vector norm){
    // normal must be normalized
    return dir - 2 * dot(dir, norm) * norm;
}

function vector vectortoline(
    vector linestart; vector lineend;
    vector point; float t;
){
    // parametric vector to line
    return (1.0 - t) * linestart + t * lineend - point;
}

function vector nearestpointonline(
    vector linestart; vector lineend;
    vector point;
){
    // snap point to closest position on line between 2 other positions
    // from John Hughes on maths stackexchange
    vector nul = {0, 0, 0};
    vector v = lineend - linestart;
    vector u = linestart - point;
    float t = - dot(v, u) / dot(v, v);
    if((0.0 <= t) && (t <= 1.0)){
        return vectortoline(linestart, lineend, nul, t);
    }
    float lena = length(vectortoline(linestart, lineend, point, 0.0));
    float lenb = length(vectortoline(linestart, lineend, point, 1.0));
    if(lena < lenb){
        return linestart;    }
    else{
        return lineend;    }
}


function void skewlinepoints(
    vector origa; vector dira;
    vector origb; vector dirb;
    // outputs
    vector pointa; vector pointb;
){

    // don't return point parametres,
    // they can be recovered trivially if needed
    vector normal = normalize(cross(dira, dirb));
    //vector normal = cross(dirb, dira);
    // get perpendicular direction between skewlines
    vector normalb = cross(dirb, normal);
    // get perpendicular displacement distance
    float displacement = dot(origb - origa, normalb);
    // get slant of dira into perpendicular direction
    float dirslant = dot(dira, normalb);
    //float dirslant = dot(normalb, dira);

    pointa = origa +  dira * displacement / dirslant;

    // same for other line
    vector normala = cross(dira, normal);
    pointb = origb + dirb * dot(origa - origb, normala) / dot(dirb, normala);
    return;
}

function void tetdirs(
    vector pos;
    // outputs
    vector tetpoints[]){
        /* after Inigo Quilez and Paulo Falcao
        find gradient of a scalar field by sampling in
        tetrahedral pattern around pos.
        tetpoints
            - array of 4 vectors corresponding to
            tet point positions
            - multiply sampled value of each point by
            its original tet vector here,
            then sum and optionally normalise

        */
        vector base = {1, -1, 0};
        append(tetpoints, base.xyy);
        append(tetpoints, base.yyx);
        append(tetpoints, base.yxy);
        append(tetpoints, base.xxx);
    }

// barycentric coords
// after Christer Ericson's Realtime Collision Detection
function float[] barytridata(vector a, b, c){
    vector v0 = b - a;
    vector v1 = c - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float denom = d00 * d11 - d01 * d01;
    return array(d00, d01, d11, denom);
}



float[] barycoords(vector a, b, c, pos){
    float result[];
    resize(result, 3);
    vector v0 = b - a;
    vector v1 = c - a;
    vector v2 = pos - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    result[1] = (d11 * d20 - d01 * d21) / denom;
    result[2] = (d00 * d21 - d01 * d20) / denom;
    result[0] = 1.0 - result[1] - result[2];
    return result;

}

function vector coercevectorhingedir(vector normala; vector normalb; vector testaxis){
    // ensure that a "hinge" axis between two normals
    // is always oriented consistently
    float dotval = dot(cross(normala, normalb), testaxis);
    if(dotval >= 0.0){
        return testaxis;
    }
    return -testaxis;

}

function vector normalfromtripoints(vector posa; vector posb; vector posc){
    return cross(posa - posb, posa - posc);
}

// void Barycentric(Point p, Point a, Point b, Point c, float &u, float &v, float &w)
// {
//     Vector v0 = b - a, v1 = c - a, v2 = p - a;
//     float d00 = Dot(v0, v0);
//     float d01 = Dot(v0, v1);
//     float d11 = Dot(v1, v1);
//     float d20 = Dot(v2, v0);
//     float d21 = Dot(v2, v1);
//     float denom = d00 * d11 - d01 * d01;
//     v = (d11 * d20 - d01 * d21) / denom;
//     w = (d00 * d21 - d01 * d20) / denom;
//     u = 1.0f - v - w;
// }



#endif
