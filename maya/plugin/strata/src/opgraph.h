#pragma once

#include "manifold.h"

#include "op.h"

/// do we need to keep entire manifolds? can we eval the whole graph live at all times?
// how does that work with inheriting values? - if an element op doesn't OVERRIDE the value, that
// just means the previous one will be used - I think that's the definition of inheritance, right?
struct StrataOpGraph {

	// assume that ops are added to vectors in order

	std::vector<StrataOp> ops;
	std::vector<StrataManifold> manifolds;

	SmallList<SmallList<StrataOp*>> generations;

	StrataManifold baseManifold; // by default empty manifold

	StrataManifold evalOpGraph() {
		StrataManifold manifold(baseManifold);
		for (StrataOp& op : ops) {
			op.eval(manifold);
		}
		return manifold;
	}

};

