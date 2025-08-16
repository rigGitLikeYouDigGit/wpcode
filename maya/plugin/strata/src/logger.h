#pragma once
#include <string>
#include <cstdarg>
#include <iostream>
#include <math.h>

#include "lib.h"

namespace strata {

    //static int _logDepth;

	struct Log {
		/* not suitable for parallel*/

        volatile static int _logDepth;

        static int hush ; // set to 1 to silence this log object

        //template<typename...T>
        //Log(T...args) {
        //    
        //    _logDepth = std::max(_logDepth, 0);
        //    for (int n = 0; n < _logDepth; n++) {
        //    }
        //    logPrint(&args...);
        //    _logDepth += 1;
        //}

        //template<typename...T>
        //void operator()(T...args) {
        //    //logPrint(args);
        //    logPrint(std::forward<T>(args)...);
        //}
        //template<typename...T>
        //void logPrint(T...args) {
        //    //DEBUGS("log depth on print: " + std::to_string(_logDepth));
        //    for (int n = 0; n < _logDepth; n++) {
        //        COUT << "  ";
        //    }
        //    logPrintToken(args...);
        //    COUT << std::endl;
        //}


        //template<typename...T>

        //using strT = std::string;

        template<typename strT>
        Log(strT arg) {

            //_logDepth = std::max(_logDepth, 0);
            //DEBUGS("log init before print:" + std::to_string(_logDepth));
            if (_logDepth) {
                if (hush) {
                    return;
                }
                COUT << '\n';
            }
            
            logPrint(arg);
            //DEBUGS("log init after print:" + std::to_string(_logDepth));
            _logDepth = _logDepth +  1;
            //DEBUGS("log init after add:" + std::to_string(_logDepth));
        }

        template<typename strT>
        void operator()(strT arg) {
            //DEBUGS("call log, depth is" + std::to_string(_logDepth));
            logPrint(arg);
            //logPrint(strT("| ") + arg);
        }
        template<typename strT>
        void logPrint(strT arg) {
            //DEBUGS("log depth on print: " + std::to_string(_logDepth));
            if (hush) {
                return;
            }
            //logPrint('\n');
            //COUT << "\n";
            //COUT << std::endl;
            //DEBUGS("before buffer print:" + std::to_string(_logDepth));
            if (_logDepth) {
            }
            COUT << '\n';
            for (int n = 0; n < _logDepth; n++) {
                //DEBUGS("printing single buffer")
                COUT << "   |";
            }

            COUT.flush();
            /*if (_logDepth) {
                COUT << "|";
            }*/
            //DEBUGS("before token print:" + std::to_string(_logDepth));
            logPrintToken(arg);
            COUT.flush();
            //DEBUGS("after token print:" + std::to_string(_logDepth));
            //COUT << std::endl;
        }
        template<typename strT>
        void logPrintToken(strT arg) {
            //std::cout << std::to_string(arg);
            //std::cout << str(arg);
            //std::cout << arg;
            //COUT << "" << arg;
            COUT << arg;
        }
        //template<typename T>
        //void logPrintToken(T arg){
        //    //std::cout << std::to_string(arg);
        //    //std::cout << str(arg);
        //    //std::cout << arg;
        //    COUT << " " << arg;
        //}


	
        ~Log() {
            _logDepth = _logDepth - 1;
            //_logDepth = std::max(_logDepth, 0);
        }

	};

}

//DEBUGS("log depth after object init" + std::to_string(strata::Log::_logDepth));\

#define LOG(msg) \
    strata::Log l( std::string(__FUNCTION__) + " "+ std::string(__FILE__) + " "  + std::to_string(__LINE__) + " LOG:");\
    l(msg);
    


