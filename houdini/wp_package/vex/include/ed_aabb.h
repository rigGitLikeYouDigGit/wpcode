
#ifndef ED_AABB_H
#define ED_AABB_H

#define AABB_POSX 0
#define AABB_POSY 1
#define AABB_POSZ 2
#define AABB_NEGX 3
#define AABB_NEGY 4
#define AABB_NEGZ 5

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
	return dir;
}

int ingestaabb(int geo; int pr; string atout; string groupprefix){
	return ingestaabb(geo, pr, atout, groupprefix, "N");
}

#endif