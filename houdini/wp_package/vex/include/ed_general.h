
#ifndef ED_GENERAL_H
#define ED_GENERAL_H 1


function int[] nearestpointsbetweengeos(int geoa, geob){
    // brute-force NxN mutual nearest point search
    // TODO: make this less trash

    int result[];

    float mind = 10000000;
    int minpt;

    for(int pt=0; pt < npoints(geoa); pt++){

        vector p = point(geoa, "P", pt);
        int npt = nearpoint(geob, p);
        vector npos = point(geob, "P", npt);
        if(length(p - npos) < mind){
            mind = length(p - npos);
            minpt = pt;
        }
    }

    int finalnpt = nearpoint(geob, vector(point(geoa, "P", minpt)));
    result = array(minpt, finalnpt);
    return result;

}



#endif
