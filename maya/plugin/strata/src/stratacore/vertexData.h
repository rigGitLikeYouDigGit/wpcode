#pragma once

#include <array>
#include <vector>
#include "element.h"

/*
SHOULD THIS BE A FIRST-CLASS STRATA ELEMENT?
for now, no - a vertex cannot exist on its own,
cannot be part of back-propagation,
and can't be created directly in strataL
*/

namespace strata {
	struct Vertex {
		/* are we actually doing houdini things in here?
		unique corner on a face -
		I think this can also link to a unique subpatch

		order of edges matches winding order of face
		*/
		int index = -1;

		/* vertex arises from intersection between 2 edges - 
		no direct relation to half-edges / spans yet.
		
		2 edges crossing create 4 vertices, 3 crossing create 12

		vertex prev/next edge is constant, interpretations can wind backwards through them

		*/
		
		std::array<int, 2> edgeIds;
		std::array<float, 2> edgeUs;
		std::array<bool, 2> edgeDirs; 


		int iPoint = -1; // intersectionPoint

		std::vector<int> faces;

		//std::array<int, 2> edgeIds(StrataManifold& manifold);
		//std::array<float, 2> edgeUs(StrataManifold& manifold);
		//std::array<bool, 2> edgeDirs(StrataManifold& manifold);

		inline int commonEdge(Vertex& otherV) {
			if (edgeIds[0] == otherV.edgeIds[0]) {
				return edgeIds[0];
			}
			if (edgeIds[1] == otherV.edgeIds[0]) {
				return edgeIds[1];
			}
			if (edgeIds[0] == otherV.edgeIds[1]) {
				return edgeIds[0];
			}
			if (edgeIds[1] == otherV.edgeIds[1]) {
				return edgeIds[1];
			}
			return -1;
		}
		
	};

	struct VertexAlongEdgeSorter { /* pass in edge along which to sort connected vertices*/
		int alongEdgeId = -1;
		StrataManifold* manifold = nullptr; /* used in integer mode */

		bool operator()(const Vertex& left, const Vertex& right);
		bool operator()(const int& leftId, const int& rightId);
	};

	struct HEdge {
		/* here we only store parametres and connectivity for edges 
		* one "half-edge" span forming part of a face border - 
		* may be shared?
		* may have an opposite?
		*/
		int index = -1;
		int edgeIndex = -1;
		std::array<float, 2> us = { 0.0, 1.0 };

		//std::unordered_map<int, 

		inline bool isFlip() {
			return us[0] > us[1];
		}
		std::array<int, 2> vtxIds(StrataManifold& manifold);

	};
}