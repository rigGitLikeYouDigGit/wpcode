#pragma once

#include <vector>
#include <unordered_set>
#include <unordered_map>

/* significantly less involved than the python version (for now)
*/



struct GraphVisitor {

	constexpr static int kDepthFirst = 0;
	constexpr static int kBreadthFirst = 1;

	struct VisitHistory {
		std::vector<std::vector<int>>& nodePaths;
		std::vector<std::unordered_set<int>>& generations;

		VisitHistory(std::vector<std::vector<int>>& nodePaths_,
			std::vector<std::unordered_set<int>>& generations_
		) :
			nodePaths(nodePaths_), generations(generations_)
		{
		}
	};


	template<
		//typename VisitPredFnT,
		typename NextIdsPredFnT,//=NextIdsPredT,
		typename ExtraT
		// HaltPredFnT=HaltPredT
	>
	void visit(
		std::vector<std::vector<int>>& nodePaths,
		std::vector<std::unordered_set<int>>& generations,
		//VisitPredFnT& visitPred,
		NextIdsPredFnT& nextIdsPred,
		ExtraT extraData = nullptr,
		//HaltPredFnT& haltPred,
		int mode = kDepthFirst

	) {
		/* we expect starting nodePaths of the form
		[ [start node id], [other start node id], ... ] etc
		- array will be populated throughout function
		to eventually record every unique path taken to each unique endpoint
		*/
		using namespace std;

		switch (mode) {
		case kDepthFirst: {
			/* DFS loop */
			deque<vector<int>> pathsToVisit(nodePaths.begin(), nodePaths.end());
			while (pathsToVisit.size()) { /* TODO: can probably handle this with indices too*/
				VisitHistory h(nodePaths, generations);
				vector<int>& currentPath = pathsToVisit.back();
				//visitPred<ExtraT>(currentPath, h, extraData);

				std::vector<int> nextNodes = nextIdsPred(
					currentPath,
					*this,
					h,
					extraData
				);
				pathsToVisit.pop_back();
				for (int n = 0; n < static_cast<int>(nextNodes.size()); n++) {
					pathsToVisit.push_back(currentPath);
					pathsToVisit.back().push_back(nextNodes[n]);

					nodePaths.push_back(pathsToVisit.back());
				}
			}
			return;
			break;
		}
		case kBreadthFirst: {
			/* BFS loop
			trying something here with paths -
			instead of the normal stack of nodes to visit, just store
			indices into the record of node paths?
			*/
			bool uniquePaths = true;

			int genStartIndex = 0;
			int genEndIndex = static_cast<int>(nodePaths.size());
			while (genStartIndex > genEndIndex) {
				//generations.emplace_back(
				//	nodePaths.begin() + genStartIndex,
				//	nodePaths.end());
				generations.emplace_back();
				//generations.back().insert(
				//	//nodePaths.begin() + genStartIndex,
				//	nodePaths.data() + genStartIndex,
				//	//nodePaths.end()._Ptr
				//	genEndIndex - genStartIndex
				//);

				/* fine goddammit I give up I have no idea what the syntax is to add an offset to begin() while
				inserting into a set, the error messages are incomprehensible and I found nothing searching*/
				for (int gI = 0; gI < (genEndIndex - genStartIndex); gI++) {
					generations.back().insert(nodePaths[gI].back());
				}

				std::unordered_set<int> newToVisit;
				for (int i = 0; i < genEndIndex - genStartIndex; i++) {
					vector<int>& nodePath = nodePaths[genStartIndex + i];
					VisitHistory h(nodePaths, generations);

					//visitPred(nodePath);
					vector<int> nextNodes = nextIdsPred(
						nodePath,
						*this,
						h,
						extraData
					);

					/* don't do any checking or logic around discarding unique/non-unique
					node paths here - leave all that to the predicate

					copy the current path, but add new node destinations to it
					*/
					nodePaths.insert(nodePaths.end(), nodePath);
					for (int n = 0; n < static_cast<int>(nextNodes.size()); n++) {
						nodePaths[genEndIndex + n].push_back(nextNodes[n]);
					}
				}
				genStartIndex = genEndIndex;
				genEndIndex = static_cast<int>(nodePaths.size()); /* probably don't need to manage separate indices here*/
			}
			return;
		}
		}
	}
};


struct NextIdsPred {
	/* return next ids for given graph element
	*/


	/* optionally pass in whole node path up to this one - last in vector*/
	template< typename ExtraT >
	std::vector<int> operator()(
		std::vector<int>& idPath,
		GraphVisitor& visitor,
		GraphVisitor::VisitHistory& history,
		ExtraT = nullptr
		) {
		/*
		idPath: vector of nodes from source, including this one

		return vector of new DIRECT destinations from this node -
		externally these will be added on to paths
		*/
		std::vector<int> result;
		return result;
	}
};

struct VisitPred {
	/* do any actual operation on node -
	* UNSURE if this should just be the same object as NextIdsPred
	*/


	/* optionally pass in whole node path up to this one - last in vector*/
	template< typename ExtraT >
	void operator()(
		std::vector<int>& idPath,
		GraphVisitor& visitor,
		GraphVisitor::VisitHistory& history,
		ExtraT = nullptr
		) {
		return;
	}
};

