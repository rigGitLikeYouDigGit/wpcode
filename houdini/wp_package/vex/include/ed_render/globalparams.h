#ifndef ED_RENDER_GLOBALPARAMS
#define ED_RENDER_GLOBALPARAMS

// param struct built on every raytracing pass

struct GlobalParams{

    int passdepth = 0;
    int seed = 0;

    int raystoscatter = 2; // number of rays to scatter from each camera collision


}

#endif
