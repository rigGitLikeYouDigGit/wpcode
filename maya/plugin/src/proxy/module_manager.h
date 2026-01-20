#pragma once
#include "module_interface.h"
#include "file_watcher.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <Windows.h>

class ModuleManager {
public:
    struct ModuleConfig {
        std::string name;              // Unique identifier
        std::string dll_path;          // Path to DLL (can be relative)
        bool auto_reload;              // Enable hot reloading
        void* init_context;            // Context passed to initialize()
    };

    ModuleManager();
    ~ModuleManager();

    // Load a module from DLL
    bool LoadModule(const ModuleConfig& config);

    // Unload a module
    bool UnloadModule(const std::string& name);

    // Reload a module (unload + load)
    bool ReloadModule(const std::string& name);

    // Get module API (returns nullptr if not loaded)
    ModuleAPI* GetModuleAPI(const std::string& name);

    // Start watching for file changes
    void StartWatching();

    // Stop watching
    void StopWatching();

    // Manual update call (for modules that have update callbacks)
    void UpdateModules(float delta_time);

    // Get list of loaded modules
    std::vector<std::string> GetLoadedModules() const;

private:
    struct LoadedModule {
        std::string name;
        std::string original_dll_path;
        std::string temp_dll_path;
        HMODULE dll_handle;
        ModuleAPI* api;
        void* init_context;
        bool auto_reload;
    };

    bool LoadModuleInternal(const std::string& name,
        const std::string& dll_path,
        void* context);
    bool UnloadModuleInternal(const std::string& name);

    void OnFileChanged(const std::string& module_name,
        const std::string& file_path,
        void* user_data);

    std::string CreateTempCopy(const std::string& source_path);
    std::string GetAbsolutePath(const std::string& path);

    std::unordered_map<std::string, LoadedModule> modules;
    std::unique_ptr<FileWatcher> file_watcher;
};