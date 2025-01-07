#ifndef ED_RENDER_RAY
#define ED_RENDER_RAY

/* raytracing requires recursive ray scattering - to reconstruct final pixel,
rays require index of 'parent' - or maybe just final pixel index

for tracing rays as points in houdini, P is ray pos, N is ray dir

if tracing IN from lights, build up LIGHT
if tracing OUT from camera, build up ABSORPTION


*/

struct Ray{
    vector pos;
    vector dir;
    vector col = {0, 0, 0};
    int pt;
    int sampleid=-1;
    int depth=0;
}

struct RayHit{
    vector pos;
    vector uvw;
    int prim;
    vector N;
    vector diffuse = {0, 0, 0};
    float roughness;

}

// ray io functions
void initray(int geo, pt) {
    setpointattrib(0, "Cd", pt, vector(set(0, 0, 0)));
    setpointattrib(0, "parent", pt, -1);
    //setpointgroup(geo, "ray", pt, 1);
    setpointgroup(geo, "outofrange", pt, 0);
    setpointattrib(geo, "sampleid", pt, pt);
    setpointattrib(geo, "depth", pt, 0);


}

Ray loadray(int geo, pt){ // load ray struct from point
    return Ray(
        point(geo, "P", pt),
        point(geo, "N", pt),
        vector(point(geo, "Cd", pt)),
        pt,
        point(geo, "sampleid", pt),
        point(geo, "depth", pt)
    );
}

void setray(int geo; Ray ray){
    // "save" ray to point, update its attributes
    setpointattrib(geo, "P", ray.pt, ray.pos);
    setpointattrib(geo, "N", ray.pt, ray.dir);
    setpointattrib(geo, "Cd", ray.pt, ray.col);
    setpointattrib(geo, "sampleid", ray.pt, ray.sampleid);

    setpointattrib(geo, "depth", ray.pt, ray.depth);


}

void setray(int geo; Ray ray; int pt){
    // "save" ray to point, update its attributes
    setpointattrib(geo, "P", pt, ray.pos);
    setpointattrib(geo, "N", pt, ray.dir);
    setpointattrib(geo, "Cd", pt, ray.col);
    setpointattrib(geo, "depth", pt, ray.depth);
    setpointattrib(geo, "sampleid", pt, ray.sampleid);

}


// only points marked as 'ray' will be cast out in render

void setactiveray(int geo; Ray ray; int active){
    setpointgroup(geo, "ray", ray.pt, active);
}
void setactiverays(int geo; Ray rays[]; int active){
    foreach(Ray ray; rays){
        setpointgroup(geo, "ray", ray.pt, active);
    }

}

void setrayend(int geo; Ray ray; int pt; float range){
    // "save" ray to point, update its attributes
    setpointattrib(geo, "P", pt, ray.pos + ray.dir * range);
    setpointattrib(geo, "N", pt, ray.dir);
    setpointattrib(geo, "Cd", pt, ray.col);
}

// ray active operations
RayHit rayintersect(int collidegeo; Ray ray; float range){
    // wrapper around normal intersect function to return RayHit struct
    RayHit hit;
    hit.prim = intersect(collidegeo, ray.pos, ray.dir * range, hit.pos, hit.uvw);
    return hit;
}

void setrayoutofrange(int geo; Ray ray; float range){
    // flag ray as terminating out of range
    setpointgroup(geo, "outofrange", ray.pt, 1);
    //setpointgroup(geo, "ray", ray.pt, 0);
    setray(geo, ray);
    setpointattrib(geo, "P", ray.pt, ray.pos + ray.dir * range);
    setactiveray(geo, ray, 0);

}


void moveraypointtohit(int geo; Ray ray; RayHit hit){
    // snap ray source point to its hit position
    setpointattrib(geo, "P", ray.pt, hit.pos);
}


void fillrayhit(int geo; RayHit hit){
    // populate RayHit with any interpolated attributes,
    // this should be only time we sample the mesh
    hit.N = primuv(geo, "N", hit.prim, hit.uvw);
    hit.diffuse = primuv(geo, "Cd", hit.prim, hit.uvw);
    hit.roughness = primuv(geo, "mat_roughness", hit.prim, hit.uvw);
}

int makepointforray(int geo; Ray ray){
    // create new point with matching normal
    int npt = addpoint(geo, ray.pos );
    ray.pt = npt;
    setray(geo, ray);
    return npt;
}

int[] makepointsforrays(int geo; Ray rays[]){
    int result[];
    resize(result, len(rays));
    for(int i=0; i<len(rays); i++){
        result[i] = makepointforray(geo, rays[i]);
    }
    return result;
}

#endif
