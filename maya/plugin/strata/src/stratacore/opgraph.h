#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../dirtyGraph.h"
#include "manifold.h"
#include "op.h"

namespace ed {
	/// do we need to keep entire manifolds? can we eval the whole graph live at all times?
	// how does that work with inheriting values and geometry? - if an element op doesn't OVERRIDE the value, that
	// just means the previous one will be used - I think that's the definition of inheritance, right?

	/* 
	* redoing to copy separate versions of the entire op graph, between maya nodes
	* 
	* if graph don't work
	* use more graph
	* 
	* 
	* to easily copy entire graphs, holding different classes of op nodes,
	* need to add functions to copy the unique_ptrs from the originals
	
	struct Base
{
	//some stuff

	auto clone() const { return std::unique_ptr<Base>(clone_impl()); }
protected:
	virtual Base* clone_impl() const = 0;
};

struct Derived : public Base
{
	//some stuff

protected:
	virtual Derived* clone_impl() const override { return new Derived(*this); };
};

struct Foo
{
	std::unique_ptr<Base> ptr;  //points to Derived or some other derived class

	//rule of five
	~Foo() = default;
	Foo(Foo const& other) : ptr(other.ptr->clone()) {}
	Foo(Foo && other) = default;
	Foo& operator=(Foo const& other) { ptr = other.ptr->clone(); return *this; }
	Foo& operator=(Foo && other) = default;
};

	*/

	struct StrataOpGraph : EvalGraph<StrataManifold>{
	};
}
