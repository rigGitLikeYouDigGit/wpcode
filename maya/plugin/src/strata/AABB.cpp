/*
  Copyright (c) 2009 Erin Catto http://www.box2d.org
  Copyright (c) 2016-2018 Lester Hedges <lester.hedges+aabbcc@gmail.com>

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.

  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  3. This notice may not be removed or altered from any source distribution.

  This code was adapted from parts of the Box2D Physics Engine,
  http://www.box2d.org
*/

#include "AABB.h"

// This file is intentionally minimal as the template implementation
// is now header-only in AABB.h for optimal performance.
// 
// Explicit template instantiations for common dimensions:

namespace aabb
{
    // Explicitly instantiate common template types to reduce compile time
    // in translation units that include this header
    template class AABB<2, float>;
    template class AABB<3, float>;
    template class AABB<2, double>;
    template class AABB<3, double>;
    template class Tree<2, float>;
    template class Tree<3, float>;
    template class Tree<2, double>;
    template class Tree<3, double>;
}