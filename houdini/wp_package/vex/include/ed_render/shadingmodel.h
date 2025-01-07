#ifndef ED_RENDER_SHADINGMODEL
#define ED_RENDER_SHADINGMODEL

#include "ed_maths.h"

#include "ed_render/globalparams.h"
#include "ed_render/ray.h"
#include "ed_render/material.h"

/*
no way to do base classes in vex -
for now shadingmodels will only use functions,
no advantage to being structs

VERY IMPORTANT that shadingmodel functions are named consistently:
mat_modelname_shadingmodel(ray, matparams, globalparams)

maybe try using BSDFs - L is light, N normal, V view

*/

MaterialParams loadmatparams(RayHit hit){

}

vector colouratinfinity(Ray ray; GlobalParams globalparams) {
    // taken from RTIOW
    float t = 0.5*(ray.dir.y + 1.0);
    vector horizoncol = {1, 1, 1};
    vector skycol = {0.5, 0.7, 1};
    return lerp(horizoncol, skycol, t) * pow(0.6, ray.depth) * pow(1.0 / globalparams.raystoscatter, ray.depth);

}


Ray[] scatterbsdf(bsdf F; Ray incident; RayHit hit; int samples, seed){
    // return scattered rays for the given bsdf

    Ray result[];
    resize(result, samples);

    for(int i=0; i < samples; i++){

        vector outdir, outcolour;
        int bsdftype = 1;
        vector2 randuv = set(rand(seed + (i + 1) + incident.pt), rand(seed + i + 2 + incident.pt));

        sample_bsdf(F, -incident.dir,
            outdir, outcolour,
            //outcolour, outdir,
            bsdftype
            ,randuv[0], randuv[1]
            //,0, 1
            );
        //outdir = eval_bsdf(ph, incident.dir)

        // sample specular / reflections
        outdir = sample_direction_cone(
                reflect(incident.dir, hit.N),
            PI/2 * hit.roughness, randuv);


        //sample diffuse
        // outdir = sample_direction_cone(
        //         hit.N,
        //     PI/2 * hit.roughness, randuv);


        result[i].dir = outdir;
        result[i].pos = hit.pos + hit.N * 0.01;
        result[i].depth = incident.depth + 1;
        result[i].sampleid = incident.sampleid;
        result[i].col = set(0, 0, 0);
    }

    return result;

}


Ray[] mat_dielectric_shadingmodel(
    Ray ray;
    RayHit hit;
    GlobalParams globalparams){
        /* normal "ubershader"-esque disney principled shader


        we return an array of new rays, for rays that will be scattered
        from point of impact
        leave empty for highly glossy surfaces, or for refraction */
        //Ray result[];

        //ray.col += hit.diffuse;

        bsdf F = diffuse(hit.N, hit.N, 0.0);

        Ray result[] = scatterbsdf(F, ray, hit,
            globalparams.raystoscatter, globalparams.seed);


        return result;
    }



#endif
