



#include "containers.h"

using namespace ed;

//// ---------------------------------------------------------------------------------
//// SmallList Implementation
//// ---------------------------------------------------------------------------------
//template <class T, unsigned int N>
//SmallList<T, N>::ListData::ListData()
//
//template <class T, unsigned int N>
//SmallList<T, N>::SmallList(void)
//
//
//template <class T, unsigned int N>
//SmallList<T, N>::SmallList(const SmallList& other)
//
//
//template <class T, unsigned int N>
//SmallList<T, N>::SmallList(const int& size)
//
//
//template <class T, unsigned int N>
//SmallList<T, N>& SmallList<T, N>::operator=(const SmallList<T, N>& other)
//{
//	SmallList(other).swap(*this);
//	return *this;
//}
//
////template <class T, unsigned int N>
////SmallList<T>& SmallList<T, N>::operator=(const SmallList<T>& other)
//
//
//template <class T, unsigned int N>
//SmallList<T, N>::~SmallList()
//
//template <class T, unsigned int N>
//int SmallList<T, N>::size() const
//
//template <class T, unsigned int N>
//T& SmallList<T, N>::operator[](int n)
//
//
//template <class T, unsigned int N>
//const T& SmallList<T, N>::operator[](int n) const
//
//
//template <class T, unsigned int N>
//int SmallList<T, N>::find_index(const T& element) const
//
//
//template <class T, unsigned int N>
//void SmallList<T, N>::clear()
//
//template <class T, unsigned int N>
//void SmallList<T, N>::reserve(int n)
//
//
//template <class T, unsigned int N>
//void SmallList<T, N>::push_back(const T& element)
//
//
//template <class T, unsigned int N>
//T SmallList<T, N>::pop_back()
//
//
//template <class T, unsigned int N>
//void SmallList<T, N>::swap(SmallList& other)
//
//
//template <class T, unsigned int N>
//T* SmallList<T, N>::data()
//
//template <class T, unsigned int N>
//const T* SmallList<T, N>::data() const


// ---------------------------------------------------------------------------------
// FreeList Implementation
// ---------------------------------------------------------------------------------
template <class T, unsigned int N>
FreeList<T, N>::FreeList() : first_free(-1)
{
}

template <class T, unsigned int N>
int FreeList<T, N>::insert(const T& element)
{
	if (first_free != -1)
	{
		const int index = first_free;
		first_free = data[first_free].next;
		data[index].element = element;
		return index;
	}
	else
	{
		FreeElement fe;
		fe.element = element;
		data.push_back(fe);
		return data.size() - 1;
	}
}

template <class T, unsigned int N>
void FreeList<T, N>::erase(int n)
{
	assert(n >= 0 && n < data.size());
	data[n].next = first_free;
	first_free = n;
}

template <class T, unsigned int N>
void FreeList<T, N>::clear()
{
	data.clear();
	first_free = -1;
}

template <class T, unsigned int N>
int FreeList<T, N>::range() const
{
	return data.size();
}

template <class T, unsigned int N>
T& FreeList<T, N>::operator[](int n)
{
	return data[n].element;
}

template <class T, unsigned int N>
const T& FreeList<T, N>::operator[](int n) const
{
	return data[n].element;
}

template <class T, unsigned int N>
void FreeList<T, N>::reserve(int n)
{
	data.reserve(n);
}

template <class T, unsigned int N>
void FreeList<T, N>::swap(FreeList& other)
{
	const int temp = first_free;
	data.swap(other.data);
	first_free = other.first_free;
	other.first_free = temp;
}


