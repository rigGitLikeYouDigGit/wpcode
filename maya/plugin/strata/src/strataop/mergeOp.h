#pragma once


#pragma once

#include "../stratacore/op.h"
#include "../stratacore/opGraph.h"
#include "../stratacore/manifold.h"

/* merges all incoming streams of manifold data
* if name is found in incoming, overwrites the base

*/

namespace ed {


	struct StrataMergeOp : StrataOp {
		/* merge input manifolds
		*/
		using StrataOp::StrataOp;

		Affine3f worldMat = Affine3f::Identity(); // matrix to tack on to transform manifold data

		virtual Status makeParams();

		///template <typename AuxT>
		virtual Status eval(StrataManifold& value, EvalAuxData* auxData, Status& s);


		virtual StrataMergeOp* clone_impl() const;

		virtual SAtomBackDeltaGroup bestFitBackDeltas(Status* s, StrataManifold& finalManifold, SAtomBackDeltaGroup& front);


	};


}


