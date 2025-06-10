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


	struct StrataOpGraph : EvalGraph<StrataManifold>{
		/* add OVERRIDE MAP of element data - 
		this will only exist for a single graph object, and serves to 
		override any element data 
		from elements created as graph moves

		does it matter that we override the entire data object? 
		maybe in the future a finer breakup of attributes somehow

		*/

		

	};
}
