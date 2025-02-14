// simple lines


#ifndef API_MACROS
#define API_MACROS 1

#include "maya/MStreamUtils.h"
#include "maya/MString.h"

// debug macros
#define COUT MStreamUtils::stdOutStream()
#define CERR MStreamUtils::stdErrorStream()

 // as in "debugString"
#define MCHECK(stat,msg)             \
        if ( MS::kSuccess != stat ) {   \
                cerr << __LINE__ << msg;            \
                return MS::kFailure;    \
        }


#define DEBUGS(info) \
COUT << info << std::endl;

#define DEBUGSL(info) \
COUT << __FILE__ << " " << __LINE__ << " \n" << info << std::endl;

// as in "debugVectorInt"
#define DEBUGVI(vec) \
for(auto const& i: vec){ \
	COUT << i << " "; \
} COUT << "length " << vec.size() << std::endl;

// as in "debugVectorInt"
#define DEBUGMVI(vec) \
for(auto const& i: vec){ \
	COUT << i << " "; \
} COUT << "length " << vec.length() << std::endl;

// as in "debugMVector"
#define DEBUGMV(vec) \
COUT << vec[0] << ", " << vec[1] << ", " << vec[2] << std::endl;

#define DEBUGVF(vec) \
copy( vec.begin(), vec.end(), ostream_iterator<float>(MStreamUtils::stdOutStream, " "));


// maths macros
#define EPS 0.000001

#define EQ(a, b) \
	(abs(a - b) < EPS)\

// to tune of twinkle twinkle little star
#define PI 3.141592653589


#endif
