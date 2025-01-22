#ifndef GROUP_H


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
