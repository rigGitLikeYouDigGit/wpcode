
#include <variant>

//#include "assignAtom.h"
//#include "callAtom.h"
//#include "constantAtom.h"
//#include "expAtom.h"
//#include "groupAtom.h"
//#include "nameAtom.h"


/* collect together all variant types for atom system
*/

namespace strata
{
	namespace expns {

		struct ExpAtom; // forward declaration
		struct PrefixParselet;
		struct InfixParselet;

		struct AssignAtom;
		struct AccessAtom;
		struct CallAtom;
		struct ConstantAtom;
		struct GetItemAtom;
		struct GreaterThanAtom;
		struct GroupAtom;
		struct LessThanAtom;
		struct PlusAtom;
		struct NameAtom;
		struct ResultAtom;

		using PrefixParseletVariant = std::variant<
			PrefixParselet,
			ConstantAtom,
			GroupAtom,
			NameAtom,

			std::monostate
		>;

		using InfixParseletVariant = std::variant<
			InfixParselet,
			AccessAtom,
			AssignAtom,
			CallAtom,
			GetItemAtom,
			GreaterThanAtom,
			
			LessThanAtom,
			PlusAtom,
			ResultAtom,
			std::monostate
		>;

	}
}
