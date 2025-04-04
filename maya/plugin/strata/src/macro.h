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

#define NODELOG(msg) \
	DEBUGSL( (MFnDependencyNode(thisMObject()).name() + ": " + msg) );

#define NODENAME (MFnDependencyNode(thisMObject()).name())


#define MNCHECK(stat, msg) \
	MCHECK(stat, "ERR - " + NODENAME + ": " + msg);

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
#define EPS_F 0.000001f

//#define INT(x) static_cast<int>(x) ///// size_t is hands down the most dumb bit of the language
//// oh um well y-you see ACKSHULLLY on some architectures the MACKSIM-M-MUM SIZE of a container might EXCEED-
/// brother just kindly shut up and go away yeah

#define EQ(a, b) \
	(abs(a - b) < EPS)\

// to tune of twinkle twinkle little star
#define PI 3.141592653589

#define seqIndex(n, limit)\
	(n >= 0 ? n : limit + n) % (limit + 1)\

#define seqContains(seq, v)\
	(std::find(seq.begin(), seq.end(), v) != seq.end())\


/* example of the macro to give to the system below
# define STRATABASE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aStInput; \
prefix MObject nodeT aStOpName; \
prefix MObject nodeT aStOutput; \
*/

// create lines of the form 'static MObject aStPoint;'
# define DECLARE_STATIC_NODE_H_MEMBERS(attrsMacro) \
attrsMacro(static, )

// create lines of the form 'MObject StrataElementOpNode::aStPoint;'
# define DEFINE_STATIC_NODE_CPP_MEMBERS(attrsMacro, nodeT) \
attrsMacro( , nodeT::)

#endif
