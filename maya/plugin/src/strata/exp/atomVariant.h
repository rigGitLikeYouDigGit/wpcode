
#pragma once
#include <variant>

// Base atom types
#include "expAtom.h"

/* collect together all variant types for atom system
*/

// Prefix parselet atoms
//#include "constantAtom.h"
//#include "groupAtom.h"
//#include "nameAtom.h"
//
//// Infix parselet atoms
//#include "assignAtom.h"
//#include "callAtom.h"
//#include "plusAtom.h"
//#include "resultAtom.h"
//#include "expElCompare.h"  // For GreaterThanAtom, LessThanAtom
//#include "accessAtom.h"
//#include "getItemAtom.h"

namespace strata
{
	namespace expns {

		struct ExpAtom; // forward declaration
		struct PrefixParselet;
		struct InfixParselet;

	/*	struct AssignAtom;
		struct AccessAtom;
		struct CallAtom;
		struct ConstantAtom;
		struct GetItemAtom;
		struct GreaterThanAtom;
		struct GroupAtom;
		struct LessThanAtom;
		struct PlusAtom;
		struct NameAtom;
		struct ResultAtom;*/

		using PrefixParseletVariant = std::variant<
			PrefixParselet
			/*ConstantAtom,
			GroupAtom,
			NameAtom,*/

			//std::monostate
		>;

		using InfixParseletVariant = std::variant<
			InfixParselet
			/*AccessAtom,
			AssignAtom,
			CallAtom,
			GetItemAtom,
			GreaterThanAtom,
			
			LessThanAtom,
			PlusAtom,
			ResultAtom,*/

			//std::monostate
		>;

		using ExpAtomVariant = std::variant<
			ExpAtom
			/*AssignAtom,
			AccessAtom,
			CallAtom,
			ConstantAtom,
			GetItemAtom,
			GreaterThanAtom,
			GroupAtom,
			LessThanAtom,
			PlusAtom,
			NameAtom,
			ResultAtom,*/
			//std::monostate
		>;

	}
}
