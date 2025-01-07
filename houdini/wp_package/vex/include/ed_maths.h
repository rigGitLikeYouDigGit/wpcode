#ifndef ED_MATHS_H
// general maths things
//#define PI 3.141592653589


#define cotan(x) (cos(x) / sin(x))
#define sec(x) (1 / cos(x))
#define cosec(x) (1 / sin(x))

function float cubicscan(float x; float centre; float width )
{
    /*
    isolate a region of a continuous signal with a cubic curve
     thanks iq <3
    */
    x = abs(x - centre);
    if( x>width ){ return 0.0;}
    x /= width;
    return 1.0 - x*x*(3.0-2.0*x);
}



#define ED_MATHS_H 1
#endif
