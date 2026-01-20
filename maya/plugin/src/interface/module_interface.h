#pragma once
#include <cstdint>

#define MODULE_API __declspec(dllexport)

// Every implementation DLL must export this interface
struct ModuleAPI {
    uint32_t version;
    const char* name;

    // Lifecycle callbacks
    bool (*initialize)(void* context);
    void (*shutdown)(void* context);
    void (*reload)(void* context);  // Called after hot reload

    // Optional update callback
    void (*update)(float delta_time);

    // Module-specific function table can extend this
    // e.g., for a render module:
    // void (*render)(void* viewport_context);
};

// Every implementation DLL must export this function
extern "C" {
    MODULE_API ModuleAPI* GetModuleAPI();
}