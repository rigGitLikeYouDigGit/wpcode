
#ifndef ED_LANDMARK_H

#include "ed_group.h"

/* functions to transfer rich data by landmark geo
main attribute is "lname"
*/

// surely we can always assume the writable geo is 0?
// too
function void transferldatapoint(int togeo, topt, fromgeo, frompt){
    string lname = point(fromgeo, "lname", frompt);
    setpointattrib(togeo, "lname", topt, lname);

    foreach(string lgrp; groupsfrompoint(fromgeo, frompt)){
        setpointgroup(togeo, lgrp, topt, 1);
    }

}


#define ED_LANDMARK_H 1
#endif
