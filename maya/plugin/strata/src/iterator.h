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
is it really worth having a base class for this, and doing
either inheritance or nasty templating to 
apply it to parent objects, logic etc

or do we just copy-paste it
it's just an iterator
*/

//struct Iterator {
//	// from internalPointers.com - thanks a ton
//	using iterator_category = std::forward_iterator_tag;
//	using difference_type = std::ptrdiff_t;
//	using value_type = int;
//	using pointer = int*;  // or also value_type*
//	using reference = int&;  // or also value_type&
//
//	pointer m_ptr;
//	Iterator(pointer ptr) : m_ptr(ptr) {}
//
//
//	reference operator*() const { return *m_ptr; }
//	pointer operator->() { return m_ptr; }
//
//	// Prefix increment
//	Iterator& operator++() {
//		m_ptr++;
//		return *this;
//	}
//
//	// Postfix increment
//	Iterator operator++(int) {
//		Iterator tmp = *this;
//		++(*this);
//		return tmp;
//	}
//
//	friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
//	friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };
//
//	Iterator begin() { return Iterator(&m_data[0]); }
//	Iterator end() { return Iterator(&m_data[200]); } // 200 is out of bounds
//};

