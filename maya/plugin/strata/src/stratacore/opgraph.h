#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../dirtyGraph.h"
#include "manifold.h"

namespace ed {
	/// do we need to keep entire manifolds? can we eval the whole graph live at all times?
	// how does that work with inheriting values and geometry? - if an element op doesn't OVERRIDE the value, that
	// just means the previous one will be used - I think that's the definition of inheritance, right?

	/* 
	* redoing to copy separate versions of the entire op graph, between maya nodes
	* 
	* if graph don't work
	* use more graph
	* 
	* 
	* to easily copy entire graphs, holding different classes of op nodes,
	* need to add functions to copy the unique_ptrs from the originals
	* 
	* each version of the graph probably need not fully evaluate - we might not even need to evaluate the 
	* whole thing until the shape node, and we need to see the final result?
	* 
	* 
	
	*/

	constexpr int SDELTAMODE_NONE = 0; // no change
	constexpr int SDELTAMODE_WORLD = 1; // direct snap to given target in worldspace
	constexpr int SDELTAMODE_LOCAL = 2; // local delta on top of original final matrix
	//constexpr int SDELTAMODE_UVN = 2; // local vector delta in UVN?


	/* worldspace snap can only act once, other wise it's an eternal pin in the graph - 
	so all of these work out to saving to space data, 
	but only affects how we gather matrices from Maya?
	maybe the mode makes no difference here
	*/

	/* UVN should be allowed 
	with separate flags?*/

	struct SPointDataDelta {
		SPointData data;
		int matrixMode = SDELTAMODE_NONE;
		int uvnMode = SDELTAMODE_NONE;
		
	};

	struct StrataOpGraph : EvalGraph<StrataManifold>{
		/* add OVERRIDE MAP of element data - 
		this will only exist for a single graph object, and serves to 
		override any element data 
		from elements created as graph moves

		does it matter that we override the entire data object? 
		maybe in the future a finer breakup of attributes somehow

		*/

		std::map<std::string, SPointDataDelta> pointOverrideMap;


	};
}
