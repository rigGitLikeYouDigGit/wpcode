// simple lines


#ifndef API_MACROS
#define API_MACROS 1

#include <vector>

#include <maya/MStreamUtils.h>
#include <maya/MString.h>
#include <maya/MGlobal.h>

// debug macros
#define COUT MStreamUtils::stdOutStream()
#define CERR MStreamUtils::stdErrorStream()

//namespace std {
//	/* apparently overloading an STL function like this is
//	some kind of capital crime */
//	string to_string(MString& s) {
//		return string(s.asChar());
//	}
//	string to_string(const char* s) {
//		return string(s);
//	}
//	string to_string(std::string s) {
//		return string(s);
//	}
//}

/* yeah for future reference this really didn't work out*/

namespace strata {
	template<typename T>
	static std::string str(T any) {
		return std::to_string(any);
	}

	//template<> std::string str<MString>(MString& s){
	//	return std::string(s.asChar());
	//}
	template<> std::string str<MString>(MString s){
		return std::string(s.asChar());
	}
	template<> std::string str<const char*>(const char* s) {
		return std::string(s);
	}

	template<> std::string str<std::string>(std::string any) {
		return any;
	}


	template<typename T>
	std::string str(std::vector<T> any) {
		std::string result = "{";
		for (T& s : any) {
			result += str(s);
		}
		result += "len:" + str(any.size()) + "}";
		return result;
	}
}

 // as in "debugString"
#define MCHECK(stat,msg)             \
        if ( MS::kSuccess != stat ) {   \
                cerr << __LINE__ << msg;            \
				MGlobal::displayError(MString(strata::str(msg).c_str())); \
                return MS::kFailure;    \
        }


#define DEBUGS(info) \
COUT << info << std::endl;

#define DEBUGSL(info) \
COUT << __FILE__ << " " << __LINE__ << " \n" << info << std::endl;

#define NODELOG(msg) \
	DEBUGSL( (MFnDependencyNode(thisMObject()).name() + ": " + msg) );
#define NODELOGT(nodeT, msg) \
	DEBUGSL( (MFnDependencyNode(nodeT::thisMObject()).name() + ": " + msg) );

#define NODENAME (MFnDependencyNode(thisMObject()).name())

#define WP_INVALID_SCALAR(v)\
	(v != v)

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

#define DEBUGMMAT(msg, mat)\
COUT << msg << "{ " << std::endl;\
COUT << std::to_string(mat[0][0]) + ", " + std::to_string(mat[0][1]) + ", " + std::to_string(mat[0][2]) + ", " + std::to_string(mat[0][3]) << std::endl;\
COUT << std::to_string(mat[1][0]) + ", " + std::to_string(mat[1][1]) + ", " + std::to_string(mat[1][2]) + ", " + std::to_string(mat[1][3]) << std::endl;\
COUT << std::to_string(mat[2][0]) + ", " + std::to_string(mat[2][1]) + ", " + std::to_string(mat[2][2]) + ", " + std::to_string(mat[2][3]) << std::endl;\
COUT << std::to_string(mat[3][0]) + ", " + std::to_string(mat[3][1]) + ", " + std::to_string(mat[3][2]) + ", " + std::to_string(mat[3][3]) << std::endl; 

#define DEBUGQuat(msg, v)\
COUT << msg << "{ " << std::to_string(v[0]) + ", " + std::to_string(v[1]) + ", " + std::to_string(v[2]) + ", " + std::to_string(v[3]) + " }" << std::endl;

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

#define EQF(a, b) \
	(abs(a - b) < EPS_F)\

// to tune of twinkle twinkle little star
#define PI 3.141592653589

#define seqIndex(n, limit)\
	(n >= 0 ? n : limit + n) % (limit + 1)\

#define seqContains(seq, v)\
	(std::find(seq.begin(), seq.end(), v) != seq.end())\

#define LERP(a, b, t)\
	b * t + (1.0 - t) * b

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
