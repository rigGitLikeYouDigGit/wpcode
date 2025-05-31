#pragma once

#include <memory>
#include <vector>

/*
these classes are intended to use the "Curiously Recurring Template Pattern"
to reuse interfaces and logic across unrelated types - 



class MyRealClonable : CloneMixin<MyRealClonable> {
};



*/

namespace ed {

	template <typename T>
	struct MixinBase {

		//inline T& this_() const { // get real reference to final templated object
		//	return static_cast<T&>(*this);
		//}

		using thisT = MixinBase<T>;
		inline thisT& this_() const { 
			return static_cast<thisT&>(*this);
		}
	};

	/*
	copying and assigning, no support for polymorphism or virtual pointer stuff.
	a class implementing this should let any OWNER be trivially copiable
	*/


#define DECLARE_DEFINE_CLONABLE_METHODS(classT)\
		classT(){\
			thisRef().initEmpty();\
		}\
classT(const classT& other) noexcept{\
			copyOther(other);\
		}\
	classT(classT&& other) noexcept{\
			takeOther(other);\
		}\
		classT& operator=(const classT& other) noexcept{\
			copyOther(other);\
			return *this;\
		}\
classT& operator=(classT&& other) noexcept{\
			takeOther(other);\
			return *this;\
		}\


	// ok can't do a common base template for mixins yet, that's fine - 
	// this is good enough for now
	template <typename T>
	struct StaticClonable// : public MixinBase<T> 
	{
		using thisT = typename StaticClonable<T>;

		// marking this const or not has big impact - has to return const reference, if marked const
		T& thisRef() {
			T& ref = static_cast<T&>(*this);
			//thisT& ref = static_cast<T&>(*this);
			return ref;
		}

		const T& thisConstRef() const {
			return static_cast<const T&>(*this);
		}

		T* thisPtr() {
			return static_cast<T*>(this);
		}

		thisT& thisTRef() {
			return *(static_cast<thisT*>(this));
		}

		using uniquePtrT = std::unique_ptr<T>;
		using sharedPtrT = std::shared_ptr<T>;


		void copyOther(const T& other) {
			// OVERRIDE this for main copying logic 
			//std::cout << "BASE copy other\n";
		}

		void takeOther(T& other) {
			/* override to do proper moving on complex types*/
			copyOther(other);
		}

		void initEmpty() {} // do setup for default empty constructor

		T* cloneNew() const {
			// add any logic here in real class
			T* p = new T(thisConstRef());
			//std::cout << "BASE after cloneNew init\n";

			//const T& ref = *this; // this seems to cause a copy?
			return p;
		}
		T cloneOnStack() { /// ???
			T result;
			result.copyOther(thisRef());
			return result;
		}
		T cloneOnStack() const { /// ???
			T result;
			result.copyOther(thisConstRef());
			return result;
		}
		inline uniquePtrT cloneUnique() const {
			return std::unique_ptr<T>(std::move(cloneNew()));
		}
		inline sharedPtrT cloneShared() const {
			return std::shared_ptr<T>(cloneNew());
		}

		static std::vector<uniquePtrT> cloneUniquePtrVector(const std::vector<uniquePtrT>& other) {
			std::vector<uniquePtrT> result;
			result.reserve(other.size());
			for (size_t i = 0; i < other.size(); i++) {
				result.push_back(other.at(i)->cloneUnique());
			}

			return result;
		}
	};


	/* all this serialisation stuff ends up being similar to the work I've already done in python - 
	serialisable might have child objects, so that's hierarchy, and they might have naming, and we might
	access them by paths, etc etc
	*/

	template <typename T>
	struct Serialisable : MixinBase<T> {
		using thisT = Serialisable<T>;

		const unsigned int bufferSizeInBytes() const {
			return 0;
		}

		
	};



	class TestClonable : public StaticClonable<TestClonable>{


	};

	
	static void testClonables() {

		TestClonable s;
		TestClonable& sRef = s.thisRef();

		//TestClonable* p = s.tThis();

		TestClonable* newClonable = s.cloneNew();


	}

}