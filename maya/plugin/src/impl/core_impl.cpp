#include "../interface/module_interface.h"
#include <iostream>
#include <ctime>

namespace core {
    int call_count = 0;

    bool Initialize(void* context) {
        std::cout << "[Core] Initialize called\n";
        call_count = 0;
        return true;
    }

    void Shutdown(void* context) {
        std::cout << "[Core] Shutdown called (total calls: " << call_count << ")\n";
    }

    void Reload(void* context) {
        std::cout << "[Core] Hot reloaded! Previous call count: " << call_count << "\n";
        // State is preserved across reloads (static variables persist)
    }

    void Update(float delta_time) {
        call_count++;
        if (call_count % 10 == 0) {
            std::cout << "[Core] Update #" << call_count
                << " (dt: " << delta_time << ")\n";
        }

        // CHANGE THIS MESSAGE AND REBUILD TO SEE HOT RELOAD
        // The module will reload automatically!
    }
}

static ModuleAPI s_api = {
    1,                      // version
    "Core",                 // name
    core::Initialize,
    core::Shutdown,
    core::Reload,
    core::Update
};

extern "C" {
    MODULE_API ModuleAPI* GetModuleAPI() {
        return &s_api;
    }
}