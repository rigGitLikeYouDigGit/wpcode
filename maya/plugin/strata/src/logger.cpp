

#include "logger.h"

using namespace strata;


volatile int Log::_logDepth = 0;
int Log::hush = 1;
int main() {

    Log l("first message");
}