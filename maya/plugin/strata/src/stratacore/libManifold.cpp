
/* splitting out more complex strata functions into this library - 
we shouldn't have to recompile the core data structure every time
we add a new way to manipulate it*/
#include "manifold.h"

#include "../exp/expParse.h"

using namespace strata;
using namespace strata::expns;

Status& elementGreaterThan(
	Status& s,
	StrataManifold& manifold,
	ExpValue& expA,
	ExpValue& expB,
	ExpValue& expOut
	/* do we guarantee this will always output a single element?
	or should it also be an expValue? since could be 
	multiple sub-elements that satisfy greater-than?
	*/
) {
	/* QUICK DIRTY sketch for now -
	this should probably
	only return some kind of UVN coordinates,
	that we can then pass into a separate function to generate
	sub-elements

	should we just overload this function for every permutation you
	can get out of the expression system?
	*/

	

	return s;
}

