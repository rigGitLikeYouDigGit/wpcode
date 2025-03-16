
/**
* from eyalroz on github
* absolute GODSEND
* 
 * Implementation of the factory pattern, based on suggestions here:
 *
 * http://stackoverflow.com/q/5120768/1593077
 *
 * and on the suggestions for corrections here:
 *
 * http://stackoverflow.com/a/34948111/1593077
 *
 */
#pragma once
#ifndef UTIL_FACTORY_H_
#define UTIL_FACTORY_H_

#include <unordered_map>
#include <exception>
#include <stdexcept>


namespace ed {

#ifndef UTIL_EXCEPTION_H_
	//using std::logic_error;
	
#endif

	template<typename Key, typename T, typename... ConstructionArgs>
	class Factory {
	public:
		using Instantiator = T * (*)(ConstructionArgs...);
	protected:
		template<typename U>
		static T* createInstance(ConstructionArgs... args)
		{
			return new U(std::forward<ConstructionArgs>(args)...);
		}
		using Instantiators = std::unordered_map<Key, Instantiator>;


		Instantiators subclassInstantiators;

	public:
		template<typename U>
		void registerClass(const Key& key)
		{
			// TODO: - Consider repeat-registration behavior.
			static_assert(std::is_base_of<T, U>::value,
				"This factory cannot register a class which is is not actually "
				"derived from the factory's associated class");
			auto it = subclassInstantiators.find(key);
			if (it != subclassInstantiators.end()) {
				//throw logic_error("Repeat registration of the same subclass in this factory.");
				throw std::invalid_argument("Repeat registration of the same subclass in this factory.");
				return;
			}
			subclassInstantiators.emplace(key, &createInstance<U>);
		}

	public:
		// TODO: Consider throwing on failure to find the instantiator
		T* produce(const Key& subclass_key, ConstructionArgs... args) const
		{
			auto it = subclassInstantiators.find(subclass_key);
			if (it == subclassInstantiators.end()) {
				return nullptr;
			}
			auto instantiator = it->second;
			return instantiator(std::forward<ConstructionArgs>(args)...);
		}

		bool canProduce(const Key& subclass_key) const {
			return subclassInstantiators.find(subclass_key) != subclassInstantiators.end();
		}
	};

} // namespace util

#endif /* UTIL_FACTORY_H_ */