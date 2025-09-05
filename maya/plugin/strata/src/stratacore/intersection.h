#pragma once

#include "element.h"
#include "../lib.h"
#include "../libEigen.h"


namespace strata {

	/* SHOULD AN INTERSECTION POINT
	JUST BE A POINT WITH DRIVERS?
	
	PROBABLY YES

	until we have a square, stick with separate intersection objects
	*/

	struct Intersection {
		static constexpr int POINT = 0;
		static constexpr int EDGE = 1;
		int type = POINT;
		int index = -1;

	};

	struct IntersectionPoint : Intersection {
		/* single point of intersection connecting multiple elements?
		* point + point (technically)
		* curve + point
		* curve + curve
		* curve + surface (not subspace)
		*/
		// should this be a map? some kind of bidirectional map?
		/* WHAT IF THE SAME EDGE CROSSES THIS POINT MULTIPLE TIMES*/

		std::unordered_map<int, std::vector<Vector3f>> elUVNMap;
		//std::map<int, Vector3f> elUVNMap;
		Vector3f pos;

		/* should this connect to vertices? not yet*/

	};

	struct IntersectionCurve : Intersection {
		/* curve region
		* curve + subcurve
		* surface + surface
		*
		* man I wish I knew a better way to structure this
		*
		*/
		std::vector<int> elements;
		std::vector<std::vector<Vector3f>> uvns;
		bez::CubicBezierPath curve;

	};


	/* each node only visited once in each generation -
	but multiple paths may lead to it*/

	/* ( u, other edge gId, intersectionPoint id )*/
	using EdgeEdgePoint = std::tuple<float, int, int>;

	struct EdgeIntersectionMap {
		/* intersection map for a single edge - only points
		*/

	};


	struct IntersectionRecord {
		/*
		* making progress - setting all maps to use indices so intersectionRecord can be copied more easily
		* although merging manifolds is still going to be a massive hassle
		*
		* we duplicate a lot of data here, once everything's working, see if we can unify with the driver structs
		*/
		std::vector<IntersectionPoint> points;
		std::vector<IntersectionCurve> curves;



		std::vector<std::vector<EdgeEdgePoint>> edgeEdgePointVecs;

		// map of {element index :  
		std::map < int,
			//std::map<Vector3i, 
			//	//IntersectionPoint*
			//	int
			//>
			Vector3iMap<int>
		> elUVNPointMap;

		/* want some way to say 'show me all elements possibly intersecting this one'
		*/
		std::map< int, // from el index
			std::map< int, // to el index
			std::vector< //separate intersections between these two elements
			//std::pair<IntersectionPoint*, IntersectionCurve*> // either point or curve
			std::pair<int, int> // dense index, type of index
			>>> elMap;

		/* TODO: can we sort elMap by uvn somehow */

		//std::map< Vector3i, IntersectionPoint* > posPointMap;
		//std::map< Vector3i, int > posPointMap;
		Vector3iMap<int> posPointMap;

		IntersectionPoint* newPoint(Vector3f v) {
			int newIdx = static_cast<int>(points.size());
			points.emplace_back();
			points[newIdx].index = newIdx;
			posPointMap[toKey(v)] = newIdx;
			points[newIdx].pos = v;
			return &points[newIdx];
		}

		IntersectionCurve* newCurve() {
			int newIdx = static_cast<int>(curves.size());
			curves.emplace_back();
			curves[newIdx].index = newIdx;
			curves[newIdx].type = Intersection::EDGE;
			return &curves[newIdx];
		}

		IntersectionPoint* getPointByVectorPosition(Vector3f worldPos, bool create = false);
		//IntersectionPoint* getPointByElUVN(int gId, Vector3f uvn, bool create = false);

		std::vector<
			std::pair<IntersectionPoint*, IntersectionCurve*>
		> getIntersectionsBetweenEls(int gIdA, int gIdB);

		std::vector<
			std::pair<IntersectionPoint*, IntersectionCurve*>
		> getIntersectionsBetweenEls(int gIdA, std::vector<int> gIdsB);

		void _sortElMap();
	};
}
