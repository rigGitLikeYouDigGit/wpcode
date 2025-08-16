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

#include "wpshared/enum.h"
#include "../macro.h"
#include "../status.h"
#include "../containers.h"
//#include "../api.h"
#include "../lib.h"
//#include "../libEigen.h"

namespace strata {


	using namespace Eigen;

	/* eventually we might use some kind of interning,
	so trying to alias name strings right now
	to make it easier to update later */
	using StrataName = std::string;


	//int a = seqIndex(-5, 3);

	// don't need full int here but MAYBE in the long run, we'll need to make an enum attribute in maya for it
		//BETTER_ENUM(SElType, int, point, edge, face);

	enum SElType : short { point, edge, face };

	struct StrataManifold;

	int constexpr stMaxParents = 3;

	int constexpr ST_TARGET_MODE_LOCAL = 0;
	int constexpr ST_TARGET_MODE_GLOBAL = 1;
	//int MATRIX_MODE_LOCAL_O

	int constexpr ST_EDGE_DENSE_NPOINTS = 10;

	//struct ExpValue;

	struct SElement {
		// whenever an element is added to manifold, during graph eval, 
		// that element is IMMUTABLE from then on, within that version of the graph
		// so this system contains absolutely no live behaviour, only static connections and positions


		StrataName name;
		SElType elType = SElType::point;
		// ANOTHER DAY, ANOTHER INABILITY TO USE CONST

		int elIndex = -1; // index within this element's type - the 3rd point, the 3rd face etc
		int globalIndex = -1; // unique global index across all elements
		//std::vector<int> drivers; // topological drivers, not parent spaces
		//std::vector<int> parents; // weighted parent influences

		//std::vector<StrataName> drivers; // topological drivers, not parent spaces
		std::vector<int> drivers; // topological drivers, not parent spaces
		/* unsure if this should be names or int ids, not sure it's so important
		*/
		std::vector<StrataName> spaces; // weighted parent influences

		std::vector<std::string> opHistory;

		/* i think we need name vectors for these instead
		*/

		std::vector<int> edges; // edges that draw from this element
		std::vector<int> points; // points that draw from this element
		std::vector<int> faces; // faces that use this edge as a rib or boundary, or pass through this point


		// todo: bit fields
		// todo: have all flags indicate a reason an element SHOULDN'T appear?
		bool isActive = true; // should this element (face) be rendered/meshed?
		bool isInvalid = false; // does this element have an error in its history?
		std::string errorMsg;

		// face attributes for winding and orientation - 
		// structural to result manifold, so put it here instead of in data
		std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
		// true for forwards, false for backwards
		bool flipFace = 0; // if face as a whole should be flipped, after edge winding

		SElement(StrataName elName, const SElType t = SElType::point) : name(elName), elType(t) {
			//name = elName;
			//elType = t;
		};


		// neighbours do not include child / driven elements
		// topology functions only return raw ints from these structs - 
		// manifold is needed to get rich pointers

		// ...should these return std::set instead? uggggh

		std::vector<int> otherNeighbourPoints(StrataManifold& manifold) {}
		std::vector<int> otherNeighbourEdges(StrataManifold& manifold) {}
		std::vector<int> otherNeighbourFaces(StrataManifold& manifold, bool requireSharedEdge = true) {}
		/// topological sets - after Keenan Crane
		// star is the set of all edges directly touching this element in it
		// for faces this might include intersectors, not sure yet
		std::vector<int> edgeStar(StrataManifold& manifold) {
		}
		// link is the set of all edges exactly one edge away from this element
		std::vector<int> edgeLink(StrataManifold& manifold) {
		}
		// do the same for other element types


		inline std::vector<int> intersectingElements(StrataManifold& manifold) {
			/* this likely will be a very wide function -
			if we want to define elements and expressions in a more topological way,
			we need a general sense of intersection between elements -

			2 edges that cross are intersecting, regardless of which one is driven
			an edge that helps define a face intersects that face -
				and the expression (face n edge) gives the sub-edge touching only the face.

			maintain an order here if possible - list edge intersections in the direction of this edge,
				list face intersections in the face's winding order, etc
			*/
		}

	};


}

