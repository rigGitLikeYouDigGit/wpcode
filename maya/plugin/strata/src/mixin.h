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

	DirtyGraph(DirtyGraph const& other) {
				copyOther(other);
			}
			DirtyGraph(DirtyGraph&& other) = default;
			DirtyGraph& operator=(DirtyGraph const& other) {
				copyOther(other);
			}
			DirtyGraph& operator=(DirtyGraph&& other) = default;

			auto clone() const { return std::unique_ptr<DirtyGraph>(clone_impl()); }
			template <typename T>
			auto clone() const { return std::unique_ptr<T>(static_cast<T*>(clone_impl())); }
			auto cloneShared() const { return std::shared_ptr<DirtyGraph>(clone_impl()); }
			template <typename T>
			auto cloneShared() const { return std::shared_ptr<T>(static_cast<T*>(clone_impl())); }
			virtual DirtyGraph* clone_impl() const {
				auto newPtr = new DirtyGraph(*this);
				newPtr->copyOther(*this);
				return newPtr;
			};

	*/


	/*thisT(thisT const& other) {
		\
			this_().copyOther(other); \
	}\*/
#define DECLARE_DEFINE_CLONABLE_METHODS(classT)\
		thisT(){\
			this2()->initEmpty();\
		}\
		thisT(thisT const& other){\
			copyOther(other);\
		}\
		thisT(thisT&& other){\
			copyOther(other);\
		}\
		classT& operator=(classT&& other){\
			copyOther(other);\
			return *this;\
		}\
		classT& operator=(const classT& other){\
			copyOther(other);\
			return *this;\
		}\
		//classT& operator=(classT const& other) {\
		//	copyOther(other);\
		//	return *this;\
		//}\



	template <typename T>
	struct StaticClonable : MixinBase<T> {
		using thisT = StaticClonable<T>;

		//StaticClonable<T>() {}
		inline const thisT& this_() const {
			//return static_cast<const thisT&>(*this);
			return *static_cast<const thisT*>(this);
		}
		inline thisT* this2() {
			return static_cast<thisT*>(this);
		}
		using uniquePtrT = std::unique_ptr<thisT>;
		using sharedPtrT = std::shared_ptr<thisT>;

		void copyOther(const thisT& other) {}

		void initEmpty() {} // do setup for default empty constructor

		DECLARE_DEFINE_CLONABLE_METHODS(thisT)


		thisT cloneOnStack() const { /// ???
			//thisT result();
			thisT result;
			//result.copyOther(this_());
			//result.copyOther(*this);
			result.copyOther(*this);
			return result;
		}

		auto cloneNew() const {
			// add any logic here in real class
			auto p = new thisT(*this);
			//p->copyOther(this_());
			p->copyOther(*this);
			return p;
		}
		inline uniquePtrT cloneUnique() const {
			return std::unique_ptr<thisT>(std::move(cloneNew()));
		}
		inline sharedPtrT cloneShared() const {
			return std::shared_ptr<thisT>(cloneNew());
		}

		static std::vector<uniquePtrT> cloneUniquePtrVector(const std::vector<uniquePtrT>& other) {
			std::vector<uniquePtrT> result;
			result.reserve(other.size());
			for (size_t i = 0; i < other.size(); i++) {
				result.push_back( other.at(i)->cloneUnique() );
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
}