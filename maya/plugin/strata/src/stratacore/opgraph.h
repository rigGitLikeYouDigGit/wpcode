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

		
		int newTemplateAttr = 0;

		virtual void copyOtherNodesVector(const StrataOpGraph& other);

		//virtual StrataOpGraph* clone_impl(bool copyAllResults) const;


		template <typename T>
		auto cloneShared(bool copyAllResults) const {
			LOG("OpGraph cloneShared");
			return std::shared_ptr<T>(
				static_cast<T*>(T::clone_impl(copyAllResults)));
		}

		virtual StrataOpGraph* clone_impl(bool copyAllResults) const {
			LOG("OpGraph clone impl");
			auto newPtr = new StrataOpGraph(*this);
			//newPtr->copyOther(*this, copyAllResults);
			return newPtr;
		}

		virtual void copyOther(const StrataOpGraph& other, bool copyAllResults = true) {
			LOG("OpGraph COPY OTHER, other nodes: " + str(other.nodes.size()))
			this->copyOtherNodesVector(other);
			nameIndexMap = other.nameIndexMap;
			_outputIndex = other._outputIndex;			//nodeDatas = other.nodeDatas;
			/* if graph is empty, it doesn't matter*/
			if (!nodes.size()) {
				return;
			}
			if (copyAllResults) {
				results = other.results;
			}
			else { // only copy result of output node
				results.clear();
				results.resize(other.results.size());
				results[getOutputIndex()] = other.results[getOutputIndex()];
			}
		}



		StrataOpGraph() {
		};
		~StrataOpGraph() = default;
		StrataOpGraph(StrataOpGraph const& other) {
			copyOther(other);
		}
		StrataOpGraph(StrataOpGraph&& other) = default;
		StrataOpGraph& operator=(StrataOpGraph const& other) {
			copyOther(other);
		}
		StrataOpGraph& operator=(StrataOpGraph&& other) = default;



	};
}
