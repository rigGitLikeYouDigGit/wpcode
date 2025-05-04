#pragma once

#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <string>
#include <memory>
#include <initializer_list>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t

#include "status.h"

/*

base class for using pointer iteration more easily - 
for now pass in functions you want to use to generate next(),
just like in python
*/

namespace ed {

	// map key iterator, from StackOverflow user Marius
	template<typename map_type>
	class key_iterator : public map_type::iterator
	{
	public:
		typedef typename map_type::iterator map_iterator;
		typedef typename map_iterator::value_type::first_type key_type;

		key_iterator(const map_iterator& other) : map_type::iterator(other) {};

		key_type& operator *()
		{
			return map_type::iterator::operator*().first;
		}
	};

	// helpers to create iterators easier:
	template<typename map_type>
	key_iterator<map_type> key_begin(map_type& m)
	{
		return key_iterator<map_type>(m.begin());
	}
	template<typename map_type>
	key_iterator<map_type> key_end(map_type& m)
	{
		return key_iterator<map_type>(m.end());
	}
	template<typename map_type>
	inline std::vector<map_type::key_type>key_sorted(map_type& m)
	{
		std::vector<map_type::key_type> sorted(key_begin(m), key_end(m));
		std::sort(sorted.begin(), sorted.end());
		return sorted;
	}

	//template<typename VT>
	struct ItParams {
		// parametre struct to inherit from for traversal logic
		//IteratorBase* parent;
		//void next();
		bool returnFirst = true;
		//virtual void doStep(IteratorBase* it); // cast inside function to right template?

		template <typename ItT>
		void doStep(ItT* it); // 'virtual is not allowed in a template definition' ok sure man whatever
		// return nullPtr when finished
	};

	struct IteratorBase {
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = int*;
		using pointer = int*;  // or also value_type*
		//using reference = valueType&;  // or also value_type&
		using reference = int*;  // or also value_type&


		pointer current;
		pointer origin;
	};

	template <typename VT>
	struct Iterator : IteratorBase{
		// from internalPointers.com - thanks a ton
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = VT*;
		using pointer = VT*;  // or also value_type*
		//using reference = valueType&;  // or also value_type&
		using reference = VT*;  // or also value_type&

		pointer current;
		pointer origin;

		std::shared_ptr<ItParams> paramPtr;

		inline void doStep() {
			// advance pointer by 1
			paramPtr.get()->doStep<Iterator<VT>>(this);
		}
		
		Iterator(pointer ptr, std::shared_ptr<ItParams> params) :
		{
			current = ptr;
			origin = ptr;
			paramPtr = params;
			if (paramPtr.get()->returnFirst) {
				doStep();
			}
		}

		reference operator*() const { return current; }
		pointer operator->() { return current; }

		// Prefix increment
		Iterator& operator++() {
			/* delegate to params for actual behaviour
			*/
			doStep();
			return *this;

		}

		// Postfix increment
		Iterator operator++(int) {
			Iterator tmp = *this;
			++(*this);
			return tmp;
		}

		friend bool operator== (const Iterator& a, const Iterator& b) { return a.current == b.current; };
		friend bool operator!= (const Iterator& a, const Iterator& b) { return a.current != b.current; };

		Iterator begin() {
			return Iterator(origin, paramPtr);
		}
		Iterator end() {
			return Iterator(nullptr, paramPtr);
		}
	};
}