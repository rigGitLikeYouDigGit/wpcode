#ifndef GROUP_H

#include "ed_defines.h"

// annoying that I can't cleanly template
function string[] pointgroupsmatching( int geo; string ex){
    string ptgrps[] = detailintrinsic( geo, "pointgroups" );
    if(!len(strip(ex))){ // if no expression passed, return all groups
        return ptgrps;
    }
    string result[] = {};
    foreach( string grp; ptgrps){
        if( match(ex, grp)){
            append(result, grp);
        }
    }
    return result;
}
function int clearpointgroups(int geo; int pt; string except[]){
    string ptgrps[] = detailintrinsic( geo, "pointgroups" );
    foreach( string grp; ptgrps){
        foreach( string lookup; except){
            if( match(lookup, grp)){
                break;
            }
            setpointgroup(geo, grp, pt, 0);
        }
    }
    return 1;
}
function int clearpointgroups(int geo; int pt; string except){
    string exceptArr[] = array(except);
    return clearpointgroups(geo, pt, exceptArr);
}
function int clearpointgroups(int geo; int pt){
    string exceptArr[] = {""};
    return clearpointgroups(geo, pt, exceptArr);
}

function string[] primgroupsmatching( int geo; string ex){
    string ptgrps[] = detailintrinsic( geo, "primitivegroups" );
    if(!len(strip(ex))){ // if no expression passed, return all groups
        return ptgrps;
    }
    string result[] = {};
    foreach( string grp; ptgrps){
        if( match(ex, grp)){
            append(result, grp);
        }
    }
    return result;
}
function int clearprimgroups(int geo; int pt; string except[]){
    string ptgrps[] = detailintrinsic( geo, "primitivegroups" );
    foreach( string grp; ptgrps){
        foreach( string lookup; except){
            if( match(lookup, grp)){
                break;
            }
            setprimgroup(geo, grp, pt, 0);
        }
    }
    return 1;
}
function int clearprimgroups(int geo; int pt; string except){
    string exceptArr[] = array(except);
    return clearprimgroups(geo, pt, exceptArr);
}
function int clearprimgroups(int geo; int pt){
    string exceptArr[] = {""};
    return clearprimgroups(geo, pt, exceptArr);
}

function string[] vertexgroupsmatching( int geo; string ex){
    string ptgrps[] = detailintrinsic( geo, "vertexgroups" );
    if(!len(strip(ex))){ // if no expression passed, return all groups
        return ptgrps;
    }
    string result[] = {};
    foreach( string grp; ptgrps){
        if( match(ex, grp)){
            append(result, grp);
        }
    }
    return result;
}
function int clearvertexgroups(int geo; int pt; string except[]){
    string ptgrps[] = detailintrinsic( geo, "vertexgroups" );
    foreach( string grp; ptgrps){
        foreach( string lookup; except){
            if( match(lookup, grp)){
                break;
            }
            setvertexgroup(geo, grp, pt, 0);
        }
    }
    return 1;
}
function int clearvertexgroups(int geo; int pt; string except){
    string exceptArr[] = array(except);
    return clearvertexgroups(geo, pt, exceptArr);
}
function int clearvertexgroups(int geo; int pt){
    string exceptArr[] = {""};
    return clearvertexgroups(geo, pt, exceptArr);
}

function string[] elgroupsmatching(int mode, geo; string mask){
    if(mode == PRIMT){
        return primgroupsmatching(geo, mask);
    }
    if(mode == POINTT){
        return pointgroupsmatching(geo, mask);
    }
    //if(mode == VERTEXT){
    return vertexgroupsmatching(geo, mask);
    //}
}

function int[] expandelgroups(int mode, geo; string mask){
    if(mode == PRIMT){
        return expandprimgroup(geo, mask);
    }
    if(mode == POINTT){
        return expandpointgroup(geo, mask);
    }
    //if(mode == VERTEXT){
        return expandvertexgroup(geo, mask);
    //}
}

function int clearelgroups(int mode; int geo; int pt; string except[]){
    if(mode == PRIMT){
        return clearprimgroups(geo, pt, except);
    }
    if(mode == POINTT){
        return clearpointgroups(geo, pt, except);
    }
    return clearvertexgroups(geo, pt, except);
}
function int clearelgroups(int mode; int geo; int pt; string except){
    string exceptArr[] = array(except);
    return clearelgroups(mode,geo, pt, exceptArr);
}
function int clearelgroups(int mode; int geo; int pt){
    string exceptArr[] = {""};
    return clearelgroups(mode, geo, pt, exceptArr);
}

function void setelgroup(int mode; int geo; string grp; int el; int val){
    if(mode == PRIMT) setprimgroup(geo, grp, el, val );
    if(mode == POINTT) setpointgroup(geo, grp, el, val );
}
function void setelgroup(int mode; int geo; string grp; int el){
    setelgroup(mode, geo, grp, el, 1);
}

function void setelsgroup(int mode; int geo; string grp; int els[]; int val){
    foreach(int el; els){
        setelgroup(mode, geo, grp, el, val);
    }
}



function string[] groupsfromprim( int geo; int primnum){
    // return all groups containing target prim
    string primgrps[] = detailintrinsic( geo, "primitivegroups");
    string result[] = {};
    foreach( string grp; primgrps){
        if( inprimgroup(0, grp, primnum)){
            append(result, grp);
        }
    }
    return result;
}



#define GROUP_H
#endif
