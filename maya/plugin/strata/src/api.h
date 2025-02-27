#pragma once

#include <vector>
#include <string>

#include "MInclude.h"
#include "macro.h"

#include "wpshared/enum.h"
namespace ed {

	// registering IDs of plugin nodes
	const unsigned int pluginPrefix = 101997;

	// common functions
	//enum BindState { off, bind, bound, live };
	BETTER_ENUM(BindState, int, off, bind, bound, live);
	BETTER_ENUM(LiveState, int, stop, playback, realtime);

	template<typename T>
	static MObject makeEnumAttr(char* name, T defaultVal, MFnEnumAttribute &fn) {
		MObject newBind;
		//int defaultValInt = defaultVal._to_integral();
		newBind = fn.create(name, name, 0);
		for (size_t index = 0; index < T::_size(); ++index) {
			fn.addField(T::_names()[index],
				T::_values()[index]);
		}

		//fn.setDefault(static_cast<int>(defaultVal));
		DEBUGS("setting enum default" << static_cast<int>(defaultVal)); // this works, extraction is janky
		fn.setDefault(int(defaultVal._to_index()));
		fn.setKeyable(true);
		fn.setHidden(false);
		return newBind;
	}

	const double defaultNeutral[3] = { 0.0, 0.0, 0.0 };

	static MObject makeVectorAttr(
		MString name, 
		MFnNumericAttribute& nFn,
		const double default[3]=defaultNeutral) 
	{
		MObject xObj = nFn.create(name + "X", name + "X",
			MFnNumericData::kDouble, default[0]);

		MObject yObj = nFn.create(name + "Y", name + "Y",
			MFnNumericData::kDouble, default[1]);

		MObject zObj = nFn.create(name + "Z", name + "Z",
			MFnNumericData::kDouble, default[2]);

		MObject parentObj = nFn.create(name, name,
			xObj, yObj, zObj);
		return parentObj;
	}

	template<typename T>
	static MObject makeEnumAttr(char* name, int defaultVal) {
		MFnEnumAttribute fn;
		return makeEnumAttr(name, defaultVal, fn);
	}

	template<typename T>
	static MObject makeEnumAttr(char* name) {
		return makeEnumAttr<T>(name, T::_from_integral(0));
	}
	template<typename T>
	static MObject makeEnumAttr(char* name, MFnEnumAttribute fn) {
		return makeEnumAttr<T>(name, T::_from_integral(0), fn);
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
		for (auto i : drivers) {
			for (auto j : driven) {
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
		for (MObject attrObj : attrs) {
			T::addAttribute(attrObj);
		}
	}

	template < typename T>
	inline void joinVectors(std::vector<T>& toExtend, std::vector<T>& toAdd) {
		toExtend.reserve(toExtend.size() + std::distance(toAdd.begin(), toAdd.end()));
		toExtend.insert(toExtend.end(), toAdd.begin(), toAdd.end());
	}

	static MStatus getAllConnectedPlugs(
		MPlug& queryPlug,
		MPlugArray& result,
		bool getSources, bool getSinks) {
		// returns nodes connected to attribute
		//DEBUGS("api.h getAllConnectedPlugs")
		MStatus s(MS::kSuccess);
		if (queryPlug.isArray()) {
		}
		else {
			queryPlug.connectedTo(result, getSources, getSinks, &s);
		}
		MCHECK(s, "could not get connected plugs in getAllConnectedPlugs()");

		return s;
	}

	static MStatus getAllConnectedPlugs(
		MObject& mainNode, MObject& plugAttr,
		MPlugArray& result,
		bool getSources, bool getSinks) {
		// returns nodes connected to attribute
		//DEBUGS("api.h getAllConnectedPlugs")
		MStatus s(MS::kSuccess);
		MFnDependencyNode dFn(mainNode);
		MPlug queryPlug(mainNode, plugAttr);
		if (queryPlug.isNull()) {
			s = MS::kFailure;
			MCHECK(s, "could not get query plug in getAllConnectedPlugs()");
		}
		return getAllConnectedPlugs(queryPlug, result, getSources, getSinks);
	}

	static MStatus getDrivingNode(
		MPlug& fromPlug, MObject& result) {
		MS s(MS::kSuccess);
		MPlugArray plugs;
		s = getAllConnectedPlugs(fromPlug, plugs, true, false);
		MCHECK(s, "plug error in getDrivingNode(), aborting");
		if (plugs.length() < 1) {
			s = MS::kFailure;
			MCHECK(s, "no connections found in getDrivingNode(), aborting");
		}
		result = plugs[0].node(&s);
		MCHECK(s, "error getting other node from plug in getDrivingNode(), aborting");
		return s;
	}

	static std::vector<MPlug> getAllChildPlugs(MPlug& parent) {
		// return depth-first list of all plugs under parent
		std::vector<MPlug> result{ parent };
		//parent.evaluateNumElements();
		if (parent.isArray()) {
			for (unsigned int i = 0; i < parent.numElements(); i++) {
				auto childResult = getAllChildPlugs(parent.elementByPhysicalIndex(i));
				joinVectors(result, childResult);
			}
		}
		else if (parent.isCompound()) {
			for (unsigned int i = 0; i < parent.numChildren(); i++) {
				auto childResult = getAllChildPlugs(parent.child(i));
				joinVectors(result, childResult);
			}
		}
		return result;
	}

	inline MStatus jumpToElement(MArrayDataHandle& hArray, int index) {
		// safer system for array plugs
		// creates index if it doesn't exist
		MStatus s;
		s = hArray.jumpToElement(index);
		if (MFAIL(s)) {
			MArrayDataBuilder builder = hArray.builder(&s);
			builder.addElement(index);
			s = hArray.set(builder);
			s = hArray.jumpToElement(index);
		}
		CHECK_MSTATUS_AND_RETURN_IT(s);
		return s;
	}


	template <typename UserNodeT>
	static MStatus castToUserNode(MObject& nodeObj, UserNodeT*& ptr) {
		// retrieve and cast full user-defined node for given MObject
		// thanks Matt
		MS s(MS::kSuccess);
		if (nodeObj.isNull()) {
			s = MS::kFailure;
			MCHECK(s, "object passed to castToUserNode() is null, aborting");
		}
		MFnDependencyNode nodeFn(nodeObj);

		// retrieve MPxNode pointer
		MPxNode* mpxPtr = nodeFn.userNode(&s);
		MCHECK(s, "failed to extract mpxNode pointer in castToUserNode(), aborting");

		// black science
		UserNodeT* sinkPtr = dynamic_cast<UserNodeT*>(mpxPtr);
		if ((sinkPtr == NULL) || (sinkPtr == nullptr)) {
			cerr << "failed dynamic cast to sink instance " << endl;
			s = MS::kFailure;
			ptr = nullptr;
		}
		MCHECK(s, "failed to dynamic_cast cast mpxNode pointer in castToUserNode(), good luck");

		ptr = sinkPtr; // set the reference to this new pointer

		return s;
	}

}