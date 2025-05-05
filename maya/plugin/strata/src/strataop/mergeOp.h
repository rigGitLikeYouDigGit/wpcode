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

		virtual Status makeParams() { return Status(); }

		///template <typename AuxT>
		virtual Status eval(StrataManifold& value, EvalAuxData* auxData, Status& s);

	};


}


