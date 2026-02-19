
#ifndef ED_AABB_H
#define ED_AABB_H

#include "ed_poly.h"

#define AABB_POSX 0
#define AABB_POSY 1
#define AABB_POSZ 2
#define AABB_NEGX 3
#define AABB_NEGY 4
#define AABB_NEGZ 5

#define DIRTOKENS {"posx", "posy",  "posz", "negx", "negy","negz"}

/* ordered such that negative direction (if present)
will always appear after positive in diagonal cases */

/* wish we could do proper bit stuff in vex*/
int dirint(vector v){
	if(dot(v, {1, 0, 0}) > 0.5){
		return AABB_POSX;
	}
	if(dot(v, {0, 1, 0}) > 0.5){
		return AABB_POSY;
	}
	if(dot(v, {0, 0, 1}) > 0.5){
		return AABB_POSZ;
	}
	if(dot(v, {-1, 0, 0}) > 0.5){
		return AABB_NEGX;
	}
	if(dot(v, {0, -1, 0}) > 0.5){
		return AABB_NEGY;
	}
	if(dot(v, {0, 0, -1}) > 0.5){
		return AABB_NEGZ;
	}
	return -1;
}

int ingestaabb(int geo; int pr; string atout; string groupprefix; string vat){
	vector v = prim(geo, vat, pr);
	int dir;
	// one-by-one matching normals
	if(dot(v, {1, 0, 0}) > 0.5){
		setprimgroup(0, groupprefix + "posx", pr, 1);
		setprimattrib(0, atout, pr, AABB_POSX);
		dir = AABB_POSX;
	}
	if(dot(v, {-1, 0, 0}) > 0.5){
		setprimgroup(0, groupprefix + "negx", pr, 1);
		setprimattrib(0, atout, pr, AABB_NEGX);
		dir = AABB_NEGX;
	}
	if(dot(v, {0, 1, 0}) > 0.5){
		setprimgroup(0, groupprefix + "posy", pr, 1);
		setprimattrib(0, atout, pr, AABB_POSY);
		dir = AABB_POSY;
	}
	if(dot(v, {0, -1, 0}) > 0.5){
		setprimgroup(0, groupprefix + "negy", pr, 1);
		setprimattrib(0, atout, pr, AABB_NEGY);
		dir = AABB_NEGY;
	}
	if(dot(v, {0, 0, 1}) > 0.5){
		setprimgroup(0, groupprefix + "posz", pr, 1);
		setprimattrib(0, atout, pr, AABB_POSZ);
		dir = AABB_POSZ;
	}
	if(dot(v, {0, 0, -1}) > 0.5){
		setprimgroup(0, groupprefix + "negz", pr, 1);
		setprimattrib(0, atout, pr, AABB_NEGZ);
		dir = AABB_NEGZ;
	}

	/* run over point pairs to find orientations for edge groups- 
	order goes XYZ and pos->neg

	*/

	int pts[] = primpoints(0, pr);
	vector ptpos[];
	for(int i = 0; i < 4; i++){
		append(ptpos, vector(point(0, "P", primpoints(0, pr)[i])));
	}

	int h = primhedge(0, pr);
	do{
		int otherprim = hedge_prim(0, hedge_nextequiv(0, h));
		vector otherv = prim(geo, vat, otherprim);
		int otherdir = dirint(otherv);
		int dirints[] = sort(array(dir, otherdir));
		string grpname = "e" + itoa(dirints[0]) + itoa(dirints[1]) 
			+ DIRTOKENS[dirints[0]] + DIRTOKENS[dirints[1]];
		setedgegroup(0, grpname, hedge_srcpoint(0, h), hedge_dstpoint(0, h), 1);
		h = hedge_next(0, h);
	}while(h != primhedge(0, pr));

	return dir;
}

int ingestaabb(int geo; int pr; string atout; string groupprefix){
	return ingestaabb(geo, pr, atout, groupprefix, "N");
}

#endif