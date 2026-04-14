

#include "expValue.h"

using namespace strata;
using namespace expns;

Status ExpValue::extend(std::initializer_list<ExpValue> vals) {
	/* flatten all given values into this one - types must match*/
	Status s;
	for (auto& el : vals) {
		if (!(el.t == t)) {
			STAT_ERROR(s, "Flattening base value of type: TODO passed different type: TODO ");
		}
	}
	for (auto& el : vals) {
		numberVals.insert(numberVals.end(), el.numberVals.begin(), el.numberVals.end());
		stringVals.insert(stringVals.end(), el.stringVals.begin(), el.stringVals.end());
	}
	return s;
}

std::string ExpValue::printInfo() {
	std::string result = "<expValue- varName:" + varName + " >";
	return result;
}