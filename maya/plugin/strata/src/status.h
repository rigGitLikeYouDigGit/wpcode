#pragma once
#include <memory>
#include <map>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>
//#include <cstdint>



struct Status {
	/* status object emulating MStatus in Maya -
	return from node eval functions and check in case graph should halt.

	A value or bool behaviour of 0 means success - anything else is an error
	so

	Status err = myFn();
	if (err){
		handle error
		}

	*/
	int val = 0; // zero is safe
	std::string msg = "";

	void addMsg(const std::string& newMsg) {
		/* add a report to this status, including line and file number*/

		// count number of tabs
		//int foundIndex = static_cast<int>(msg.rfind("\n"));
		//std::string pool = msg.substr(foundIndex);
		//std::string newLine("/n")
		const char* newLine = "\n";
		const char* tab = "\t";
		std::string pool = msg.substr(msg.rfind(newLine));
		//std::string::difference_type foundCount = std::count(pool.begin(), pool.end(), "\t");
		//std::string::difference_type foundCount = std::count(pool.begin(), pool.end(), tab);

		/////// COULD NOT FIGURE OUT how to count tab characters in a string, both these give errors
		int tabCount = 0;
		for (auto c : pool) {
			if (c == tab[0]) {
				tabCount += 1;
			}
		}


		int currentDepth = static_cast<int>(tabCount);
		msg.append("\n");
		for (int i = 0; i < currentDepth; i++) {
			msg.append("\t");
		}
		msg.append(__FILE__);
		msg.append(std::to_string(__LINE__));
		msg.append(newMsg);
	}

	explicit operator bool()
	{
		return (val ? true : false);
	}

};

// raise an early error from a function with value and message
#define STAT_ERROR_VAL(s, errVal, strMsg)\
	s.val = errVal; s.addMsg(strMsg); return s;

#define STAT_ERROR(s, strMsg) STAT_ERROR_VAL(s, 1, strMsg)

// for check-return status
#define	CRSTAT(s)\
	if(s){ return s; }\

#define	CRMSG(s, strMsg)\
	if(s){ s.addMsg(strMsg); return s; }\

	// for check-write
#define CWSTAT(s)\
	if(s){ DEBUGS(s.msg);}\

// check-write-message
#define CWMSG(s, strMsg)\
	if(s){ s.addMsg(strMsg); DEBUGS(s.msg);}


// check-write-return
#define CWRSTAT(s, strMsg)\
	if(s){ s.addMsg(strMsg); DEBUGS(s.msg); return s;}
