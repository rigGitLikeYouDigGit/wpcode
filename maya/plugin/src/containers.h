#pragma once

// small and efficient containers given by DragonEnergy on SO
// I don't know who you are, but I'm in your debt
#ifndef _CONTAINERS_LIB
#define _CONTAINERS_LIB

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>


// ************************************************************************************
// SmallList.hpp
// ************************************************************************************


namespace strata {
	
	typedef uint16_t uShort;
	//#define USNULL uShort(1 << 16)
	typedef unsigned int uint;

	// used for returning small arrays from functions
	template <typename T>
	struct ARR {
		T* start;
		uShort length;
	};

	// Stores a random-access sequence of elements similar to vector, but avoids
	// heap allocations for small lists. T must be trivially constructible and
	// destructible.
	template <class T, unsigned int N=4>
	class SmallList
	{
	public:
		// Creates an empty list.
		SmallList()
		{
		}

		// Creates a copy of the specified list.
		SmallList(const SmallList& other) {
			if (other.ld.cap == fixed_cap)
			{
				ld = other.ld;
				ld.data = ld.buf;
			}
			else
			{
				reserve(other.ld.num);
				for (int j = 0; j < other.size(); ++j)
					ld.data[j] = other.ld.data[j];
				ld.num = other.ld.num;
				ld.cap = other.ld.cap;
			}
		}

		// Creates SmallList and reserves given number of entries.
		SmallList(const int& size) {
			reserve(size);
		}

		// Copies the specified list.
		SmallList& operator=(const SmallList& other) {
			SmallList(other).swap(*this);
			return *this;
		}

		// Destroys the list.
		~SmallList()
		{
			if (ld.data != ld.buf) {
				free(ld.data);
			}
		}

		// Returns the number of agents in the list.
		int size() const {
			return ld.num;
		}


		// Returns the nth element.
		T& operator[](int n) {	// allow negative indexing
			assert(n >= 0 && n < ld.num);
			return ld.data[n];
		}

		// Returns the nth element in the list.
		const T& operator[](int n) const {
			assert(n >= 0 && n < ld.num);
			return ld.data[n];
		}
		// with modulo for negative indexing (non-const)
		T& at(int n) {
			n = (n >= 0 ? n : size() + n) % (size() + 1);
			assert(n >= 0 && n < ld.num);
			return ld.data[n];
		}

		// with modulo for negative indexing (const)
		const T& at(int n) const {
			n = (n >= 0 ? n : size() + n) % (size() + 1);
			assert(n >= 0 && n < ld.num);
			return ld.data[n];
		}
		// Returns an index to a matching element in the list or -1
		// if the element is not found.
		int find_index(const T& element) const {
			for (int j = 0; j < ld.num; ++j)
			{
				if (ld.data[j] == element)
					return j;
			}
			return -1;
		}

		// Clears the list.
		void clear()
		{
			ld.num = 0;
		}

		// Reserves space for n elements.
		void reserve(int n) {
			enum { type_size = sizeof(T) };
			if (n > ld.cap)
			{
				if (ld.cap == fixed_cap)
				{
					ld.data = static_cast<T*>(malloc(n * type_size));
					memcpy(ld.data, ld.buf, sizeof(ld.buf));
				}
				else
					ld.data = static_cast<T*>(realloc(ld.data, n * type_size));
				ld.cap = n;
			}
		}

		// Inserts an element to the back of the list.
		void push_back(const T& element) {
			if (ld.num >= ld.cap)
				reserve(ld.cap * 2);
			ld.data[ld.num++] = element;
		}

		/// Pops an element off the back of the list.
		T pop_back() {
			return ld.data[--ld.num];
		}

		//// add element to front of list
		//void push(const T& element) 

		//// pop element from front of list
		//T pop()


		// Swaps the contents of this list with the other.
		void swap(SmallList& other) {
			ListData& ld1 = ld;
			ListData& ld2 = other.ld;

			const int use_fixed1 = ld1.data == ld1.buf;
			const int use_fixed2 = ld2.data == ld2.buf;

			const ListData temp = ld1;
			ld1 = ld2;
			ld2 = temp;

			if (use_fixed1)
				ld2.data = ld2.buf;
			if (use_fixed2)
				ld1.data = ld1.buf;
		}

		// Returns a pointer to the underlying buffer.
		T* data() {
			return ld.data;
		}

		// Returns a pointer to the underlying buffer.
		const T* data() const {
			return ld.data;
		}

		T* begin() { return &data()[0]; }
		T* end() { return &data()[size()]; }
		T* back() { return &data()[size() - 1]; }
		const T* begin() const { return data(); }
		const T* end() const { return data() + size(); }
		const T* back() const { return data() + size() - 1; }
		T* rbegin() { return end() - 1; }
		T* rend() { return begin() - 1; }

		static const unsigned int MAXSIZE = N;

		// Add move constructor
		SmallList(SmallList&& other) noexcept {
			if (other.ld.data == other.ld.buf) {
				// Other uses inline buffer - copy it
				ld = other.ld;
				ld.data = ld.buf;
			}
			else {
				// Other uses heap - steal pointer
				ld.data = other.ld.data;
				ld.num = other.ld.num;
				ld.cap = other.ld.cap;

				// Reset other to inline buffer
				other.ld.data = other.ld.buf;
				other.ld.num = 0;
				other.ld.cap = fixed_cap;
			}
		}

		// Add move assignment
		SmallList& operator=(SmallList&& other) noexcept {
			if (this != &other) {
				// Clean up current data
				if (ld.data != ld.buf) {
					free(ld.data);
				}

				// Move from other
				*this = SmallList(std::move(other));
			}
			return *this;
		}

	private:
		enum { fixed_cap = N };
		struct ListData
		{
			ListData() : data(buf), num(0), cap(fixed_cap)
			{
			}
			T buf[fixed_cap];
			T* data;
			int num;
			int cap;
		};
		ListData ld;
	};

	/// Provides an indexed free list with constant-time removals from anywhere
	/// in the list without invalidating indices. T must be trivially constructible
	/// and destructible.
	template <class T, unsigned int N = 4>
	class FreeList
	{
	public:
		/// Creates a new free list.
		FreeList();

		/// Inserts an element to the free list and returns an index to it.
		int insert(const T& element);

		// Removes the nth element from the free list.
		void erase(int n);

		// Removes all elements from the free list.
		void clear();

		// Returns the range of valid indices.
		int range() const;

		// Returns the nth element.
		T& operator[](int n);

		// Returns the nth element.
		const T& operator[](int n) const;

		// Reserves space for n elements.
		void reserve(int n);

		// Swaps the contents of the two lists.
		void swap(FreeList& other);

	private:
		union FreeElement
		{
			T element;
			int next;
		};
		SmallList<FreeElement, N> data;
		int first_free;
	};

	template<typename T>
	struct CPair {
		/* pair storing 2 spans of the same type contiguously */
		T* data = nullptr;
		size_t first_size = 0;
		size_t second_size = 0;

		// Default constructor
		CPair() = default;

		// Constructor with allocation
		CPair(size_t first_count, size_t second_count) {
			allocate(first_count, second_count);
		}

		// Copy constructor
		CPair(const CPair& other) {
			if (other.data) {
				allocate(other.first_size, other.second_size);
				std::memcpy(data, other.data, (first_size + second_size) * sizeof(T));
			}
		}

		// Copy assignment
		CPair& operator=(const CPair& other) {
			if (this != &other) {
				if (other.data) {
					allocate(other.first_size, other.second_size);
					std::memcpy(data, other.data, (first_size + second_size) * sizeof(T));
				}
				else {
					delete[] data;
					data = nullptr;
					first_size = 0;
					second_size = 0;
				}
			}
			return *this;
		}

		// Move constructor
		CPair(CPair&& other) noexcept
			: data(other.data)
			, first_size(other.first_size)
			, second_size(other.second_size)
		{
			other.data = nullptr;
			other.first_size = 0;
			other.second_size = 0;
		}

		// Move assignment
		CPair& operator=(CPair&& other) noexcept {
			if (this != &other) {
				delete[] data;

				data = other.data;
				first_size = other.first_size;
				second_size = other.second_size;

				other.data = nullptr;
				other.first_size = 0;
				other.second_size = 0;
			}
			return *this;
		}

		void allocate(size_t first_count, size_t second_count) {
			delete[] data;
			first_size = first_count;
			second_size = second_count;
			data = new T[first_size + second_size];
		}

		T* first() { return data; }
		T* second() { return data + first_size; }

		const T* first() const { return data; }
		const T* second() const { return data + first_size; }

		size_t total_size() const { return first_size + second_size; }

		~CPair() { delete[] data; }
	};

} // /strata


#endif