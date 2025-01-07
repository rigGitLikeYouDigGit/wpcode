
#ifndef ED_RENDER_MATERIAL
#define ED_RENDER_MATERIAL 1

/*
common struct for defining material parametres
includes shadingmodel attribute, which governs dynamic dispatch
*/


struct MaterialParams{

    string name = "name";
    float opacity;

    string shadingmodel = "dielectric"; // normal pbr shading


}



#endif
