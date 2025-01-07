

#ifndef ED_DEFINES_H
#define ED_DEFINES_H 1


#define NAME( input, pt) ( point(input, "name", pt) )
#define ID( input, pt) ( point(input, "id", pt) )
#define SETPOINT(attr, pt, val) (setpointattrib(0, attr, pt, val))
#define SETPRIM(attr, pr, val) (setpointattrib(0, attr, pr, val))



#endif
