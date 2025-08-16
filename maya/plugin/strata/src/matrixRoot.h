#pragma once

/* 
adapted from Jason Weber's incredible FreeElectron project
nvm, Eigen actually includes matrix power stuff 


Copyright (c) 2003-2021, Free Electron Organization
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <maya/MVector.h>
#include <maya/MMatrix.h>


/*
for now ignore the templating in the original
*/
namespace strata {


	// NOTE fraction: 23 bits for single and 52 for double
	/**************************************************************************//**
		@brief solve B = A^^power, where A is a matrix

		@ingroup geometry

		The power can be any arbitrary real number.

		Execution time is roughly proportional to the number of set bits in
		the integer portion of the floating point power and a fixed number
		of iterations for the fractional part.

		The number of iterations used to compute of the fractional portion
		of the power can be changed.  The maximum error after each iteration
		is half of the previous iteration, starting with one half.  The entire
		integer portion of the power is always computed.
	*//***************************************************************************/
	template <typename MATRIX=MMatrix>
	struct MatrixPower
	{
	public:
		MatrixPower(void) :
			m_iterations(16) {}

		template <typename T>
		void	solve(MATRIX& B, const MATRIX& A, T a_power) const;

		void	setIterations(U32 iterations) { m_iterations = iterations; }

	private:
		MatrixSqrt<MATRIX>	m_matrixSqrt;
		U32					m_iterations;
	};

	template <typename MATRIX>
	template <typename T>
	inline void MatrixPower<MATRIX>::solve(MATRIX& B, const MATRIX& A,
		T a_power) const
	{
		T absolute = a_power;

		const bool inverted = (absolute < 0.0);
		if (inverted)
		{
			absolute = -absolute;
		}

		/*U32 whole = U32(absolute);
		F32 fraction = absolute - whole;*/
		int whole = int(absolute);
		float fraction = absolute - whole;

#if MRP_DEBUG
		feLog("\nwhole=%d\nfraction=%.6G\n", whole, fraction);
#endif

		//MATRIX R;
		//setIdentity(R);
		MATRIX R = MMatrix::identity; //TODO: obviously won't work for other matrix types

		MATRIX partial = A;
		float contribution = 1.0;
		int iteration;
		for (iteration = 0; iteration < m_iterations; iteration++)
		{
			m_matrixSqrt.solve(partial, partial);
			contribution *= 0.5;
#if MRP_DEBUG
			feLog("\ncontribution=%.6G\nfraction=%.6G\n", contribution, fraction);
#endif
			if (fraction >= contribution)
			{
				R *= partial;
				fraction -= contribution;
			}
		}

		partial = A;
		while (whole)
		{
#if MRP_DEBUG
			feLog("\nwhole=%d\n", whole);
#endif
			if (whole & 1)
			{
				R *= partial;
			}
			whole >>= 1;
			if (whole)
			{
				partial *= partial;
			}
		}

#if MRP_VALIDATE
		BWORD invalid = FALSE;
		for (U32 m = 0; m < width(R); m++)
		{
			for (U32 n = 0; n < height(R); n++)
			{
				if (FE_INVALID_SCALAR(R(m, n)))
				{
					invalid = TRUE;
				}
			}
		}
		if (invalid)
		{
			feLog("MatrixPower< %s >::solve invalid results power=%.6G\n",
				FE_TYPESTRING(MATRIX).c_str(), a_power);
			feLog("\nA\n%s\n", print(A).c_str());
			feLog("\nB\n%s\n", print(R).c_str());

			feX("MatrixPower<>::solve", "invalid result");
		}
#endif

		if (inverted)
		{
			invert(B, R);
		}
		else
		{
			B = R;
		}
	}

	struct MMatrixPower : MatrixPower<MMatrix> {};


}

