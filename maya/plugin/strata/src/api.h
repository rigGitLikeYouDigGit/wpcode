#pragma once

#include <vector>


#include <maya/MObject.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MString.h>
#include "wpshared/enum.h"
namespace ed {

	// registering IDs of plugin nodes
	const unsigned int pluginPrefix = 101997;

	// common functions
	//enum BindState { off, bind, bound, live };
	BETTER_ENUM(BindState, int, off, bind, bound, live);
	BETTER_ENUM(LiveState, int, stop, playback, realtime);

	template<typename T>
	static MObject makeEnumAttr(char* name, T defaultVal) {
		MObject newBind;
		MFnEnumAttribute fn;
		//int defaultValInt = defaultVal._to_integral();
		newBind = fn.create(name, name, 0);
		for (size_t index = 0; index < T::_size(); ++index) {
			fn.addField(T::_names()[index],
				T::_values()[index]);
		}

		fn.setDefault(static_cast<int>(defaultVal));
		DEBUGS("setting enum default" << static_cast<int>(defaultVal)); // this works, extraction is janky
		//fn.setDefault((defaultVal._to_index()));
		fn.setKeyable(true);
		fn.setHidden(false);
		return newBind;
	}

	template<typename T>
	static MObject makeEnumAttr(char* name) {

		return makeEnumAttr<T>(name, T::_from_integral(0));
	}

	static MObject makeBindAttr(char* name) {
		MObject newBind;
		MFnEnumAttribute fn;
		newBind = fn.create(name, name, 0);
		for (size_t index = 0; index < BindState::_size(); ++index) {
			fn.addField(BindState::_names()[index],
				BindState::_values()[index]);
		}
		fn.setKeyable(true);
		fn.setHidden(false);
		return newBind;
	}

	template <typename T>
	inline void setAttributeAffectsAll(MObject& driver, std::vector<MObject>& driven) {
		// sets driver to affect all driven
		for (auto& i : driven) {
			T::attributeAffects(driver, i);
		}
	}

	template <typename T>
	inline void setAttributesAffect(std::vector<MObject>& drivers, std::vector<MObject>& driven) {
		// sets driver to affect all driven
		for (auto& i : drivers) {
			for (auto& j : driven) {
				T::attributeAffects(i, j);
			}
		}
	}

	template <typename T>
	inline void setAttributesAffect(std::vector<MObject>& drivers, MObject& driven) {
		// sets driver to affect all driven
		for (auto& i : drivers) {
			T::attributeAffects(i, driven);
		}
	}

	template <typename T>
	inline void addAttributes(std::vector<MObject>& attrs) {
		// adds all attributes at once
		for (auto& attrObj : attrs) {
			T::addAttribute(attrObj);
		}
	}

}