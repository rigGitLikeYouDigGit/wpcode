
#ifndef _ED_DYNAMIC_H

#define _ED_DYNAMIC_H
#include "array.h"
#include "ed_poly.h"
#include "ed_group.h"
#include "ed_vector.h"

function vector integrateverlet(
    vector pos0; vector pos1; vector acceleration; float dt){
        // compute pos2 from verlet integration of
        // previous 2 positions
        // only returns new position, recover velocity outside
        return 2 * pos1 - pos0 + acceleration * dt * dt * 0.5;
    }

function vector[] integrateparticle(
    vector pos0; vector pos1; vector force; float mass; float dt){
        // integrate with respect to mass and force
        return integrateverlet(
            pos0, pos1, force / mass, dt
        );
    }



#endif
