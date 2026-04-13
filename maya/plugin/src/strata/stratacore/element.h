#pragma once
#include <memory>
#include <map>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
//#include <cstdint>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>

#include <maya/MVector.h>
#include <maya/MMatrix.h>

#include "../bezier/bezier.h"

//#include <enum.h>
#include "../../enum.h"
#include "../macro.h"
#include "../status.h"
#include "../../containers.h"
//#include "../api.h"
#include "../lib.h"
#include "types.h"
//#include "../libEigen.h"

#include "../name.h"

namespace strata {

	/* fundamental types : point, edge, face
	
	fundamental unit is the strataCoord - fat struct to save a consistent position in domain space or on anchor.
	same kind of data saved on creator node as regenerated.

	point needs one, edge and face needs multiple -
	coord values interpreted differently based on type of anchor
	*/

	using namespace Eigen;

	// maybe add vertex, maybe add intersection point
	BETTER_ENUM(SElType, short, point, edge, face);

	//enum SElType : short { point, edge, face, subFace }; 


	int constexpr stMaxDomains = 3;

	int constexpr ST_TARGET_MODE_LOCAL = 0;
	int constexpr ST_TARGET_MODE_GLOBAL = 1;
	//int MATRIX_MODE_LOCAL_O

	int constexpr ST_EDGE_DENSE_NPOINTS = 10;

	//struct ExpValue;

	struct SCoord {
		/* struct for a single coordinate on an anchor - get tangent, normal and twist
		vectors for curve frame
		keep up spirit of original HTTYD system, process lengths in 3 blends:
		value,
		arc length/param,
		normalised/not normalised,
		from start/from end
		*/
		short globalIndex = -1; // index of domain element
		Eigen::Vector3f uvn = { 0, 0, 0 }; // uvn coords of domain element to sample for points on this edge
		Eigen::Vector3f mode = { 0, 0, 0 }; // metadata for modes of operation (length, normalisation, direction etc)
		
		// no idea if final matrix should go here
		//AffineCompact3f finalMat = AffineCompact3f::Identity(); // final matrix for this coord, 
		// independent of subsequent elements
	};

	struct SElement {
		// whenever an element is added to manifold, during graph eval, 
		// that element is IMMUTABLE from then on, within that version of the graph
		// so this system contains absolutely no live behaviour, only static connections and positions


		StrataName name;
		SElType elType = SElType::point;

		int elIndex = -1; // index within this element's type - the 3rd point, the 3rd face etc
		int globalIndex = -1; // unique global index across all elements
		int subIndex = -1; // only for subfaces for now

		SmallList<SCoord> anchors; // topological anchors, not domain spaces
		SmallList<AffineCompact3f> anchorMats; // anchor mats in worldspace
		
		SmallList<SCoord> domains; // domain parent space

		//std::vector<std::string> opHistory; // save at manifold/graph level
		//std::set<std::string> groups; // 

		std::vector<int> edges; // edges that draw from this element
		std::vector<int> points; // points that draw from this element
		std::vector<int> faces; // faces that use this edge as a rib or boundary, or pass through this point


		// todo: bit fields
		// todo: have all flags indicate a reason an element SHOULDN'T appear?
		bool isActive = true; // should this element (face) be rendered/meshed?
		bool isInvalid = false; // does this element have an error in its history?
		bool flipFace = 0; // if face as a whole should be flipped, after edge winding

		std::string errorMsg;

		// face attributes for winding and orientation -  
		// structural to result manifold, so put it here instead of in data
		//std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
		// derive this instead from coords
		// true for forwards, false for backwards
		SElement() = default;

		SElement(StrataName elName, const SElType t = SElType::point) : name(elName), elType(t) {
			//name = elName;
			//elType = t;
		};

		bool operator==(const SElement& other) const {
			return name == other.name;
		}

		/*  T O D O :
		intersection methods below aren't useful, we know now the complex vertex stuff 
		needed to find proper rings, boundaries etc

		this object can PROBABLY be reduced to 
		- name
		- type
		- global index
		- elIndex

		- anchors
		*/


		// neighbours do not include child / driven elements
		// topology functions only return raw ints from these structs - 
		// manifold is needed to get rich pointers

		// ...should these return std::set instead? uggggh

		//std::vector<int> otherNeighbourPoints(StrataManifold& manifold) {}
		//std::vector<int> otherNeighbourEdges(StrataManifold& manifold) {}
		//std::vector<int> otherNeighbourFaces(StrataManifold& manifold, bool requireSharedEdge = true) {}
		/// topological sets - after Keenan Crane
		// star is the set of all edges directly touching this element in it
		// for faces this might include intersectors, not sure yet
		//std::vector<int> edgeStar(StrataManifold& manifold) {
		//}
		//// link is the set of all edges exactly one edge away from this element
		//std::vector<int> edgeLink(StrataManifold& manifold) {
		//}
		// do the same for other element types

	};


}

