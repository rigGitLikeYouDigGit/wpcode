#include "module_manager.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

ModuleManager::ModuleManager()
    : file_watcher(std::make_unique<FileWatcher>())
{
}

ModuleManager::~ModuleManager() {
    StopWatching();

    // Unload all modules
    auto module_names = GetLoadedModules();
    for (const auto& name : module_names) {
        UnloadModule(name);
    }
}

std::string ModuleManager::GetAbsolutePath(const std::string& path) {
    char abs_path[MAX_PATH];
    GetFullPathNameA(path.c_str(), MAX_PATH, abs_path, NULL);
    return std::string(abs_path);
}

std::string ModuleManager::CreateTempCopy(const std::string& source_path) {
    char temp_path[MAX_PATH];
    char temp_file[MAX_PATH];

    GetTempPathA(MAX_PATH, temp_path);

    // Create unique temp filename based on source
    fs::path source(source_path);
    std::string temp_name = source.stem().string() + "_" +
        std::to_string(GetTickCount64()) + ".dll";

    std::string temp_full = std::string(temp_path) + temp_name;

    if (!CopyFileA(source_path.c_str(), temp_full.c_str(), FALSE)) {
        return "";
    }

    return temp_full;
}

bool ModuleManager::LoadModule(const ModuleConfig& config) {
    if (modules.find(config.name) != modules.end()) {
        std::cerr << "Module '" << config.name << "' already loaded\n";
        return false;
    }

    std::string abs_path = GetAbsolutePath(config.dll_path);

    if (!LoadModuleInternal(config.name, abs_path, config.init_context)) {
        return false;
    }

    auto& module = modules[config.name];
    module.auto_reload = config.auto_reload;
    module.original_dll_path = abs_path;

    // Add to file watcher if auto-reload enabled
    if (config.auto_reload && file_watcher) {
        FileWatcher::WatchConfig watch_config;
        watch_config.file_path = abs_path;
        watch_config.module_name = config.name;
        watch_config.user_data = nullptr;

        file_watcher->AddWatch(watch_config);
    }

    std::cout << "Module '" << config.name << "' loaded from: " << abs_path << "\n";
    return true;
}

bool ModuleManager::LoadModuleInternal(const std::string& name,
    const std::string& dll_path,
    void* context) {
    // Create temp copy to allow rebuilding while loaded
    std::string temp_path = CreateTempCopy(dll_path);
    if (temp_path.empty()) {
        std::cerr << "Failed to create temp copy of: " << dll_path << "\n";
        return false;
    }

    // Load the DLL
    HMODULE dll = LoadLibraryA(temp_path.c_str());
    if (!dll) {
        std::cerr << "Failed to load DLL: " << temp_path
            << " (Error: " << GetLastError() << ")\n";
        DeleteFileA(temp_path.c_str());
        return false;
    }

    // Get the API
    auto GetAPI = (ModuleAPI * (*)())GetProcAddress(dll, "GetModuleAPI");
    if (!GetAPI) {
        std::cerr << "DLL missing GetModuleAPI export: " << temp_path << "\n";
        FreeLibrary(dll);
        DeleteFileA(temp_path.c_str());
        return false;
    }

    ModuleAPI* api = GetAPI();
    if (!api) {
        std::cerr << "GetModuleAPI returned null\n";
        FreeLibrary(dll);
        DeleteFileA(temp_path.c_str());
        return false;
    }

    // Initialize the module
    if (api->initialize && !api->initialize(context)) {
        std::cerr << "Module initialize failed\n";
        FreeLibrary(dll);
        DeleteFileA(temp_path.c_str());
        return false;
    }

    // Store module info
    LoadedModule module;
    module.name = name;
    module.original_dll_path = dll_path;
    module.temp_dll_path = temp_path;
    module.dll_handle = dll;
    module.api = api;
    module.init_context = context;

    modules[name] = module;
    return true;
}

bool ModuleManager::UnloadModule(const std::string& name) {
    return UnloadModuleInternal(name);
}

bool ModuleManager::UnloadModuleInternal(const std::string& name) {
    auto it = modules.find(name);
    if (it == modules.end()) {
        return false;
    }

    LoadedModule& module = it->second;

    // Call shutdown
    if (module.api && module.api->shutdown) {
        module.api->shutdown(module.init_context);
    }

    // Unload DLL
    if (module.dll_handle) {
        FreeLibrary(module.dll_handle);
    }

    // Delete temp file
    if (!module.temp_dll_path.empty()) {
        // May fail if still locked, that's ok
        DeleteFileA(module.temp_dll_path.c_str());
    }

    // Remove from file watcher
    if (file_watcher && module.auto_reload) {
        file_watcher->RemoveWatch(name);
    }

    modules.erase(it);

    std::cout << "Module '" << name << "' unloaded\n";
    return true;
}

bool ModuleManager::ReloadModule(const std::string& name) {
    auto it = modules.find(name);
    if (it == modules.end()) {
        return false;
    }

    // Save config before unloading
    std::string dll_path = it->second.original_dll_path;
    void* context = it->second.init_context;

    std::cout << "Reloading module '" << name << "'...\n";

    // Unload old version
    if (!UnloadModuleInternal(name)) {
        return false;
    }

    // Small delay to ensure file handles are released
    Sleep(50);

    // Load new version
    if (!LoadModuleInternal(name, dll_path, context)) {
        std::cerr << "Failed to reload module '" << name << "'\n";
        return false;
    }

    // Call reload callback if provided
    auto& module = modules[name];
    if (module.api && module.api->reload) {
        module.api->reload(context);
    }

    std::cout << "Module '" << name << "' reloaded successfully\n";
    return true;
}

ModuleAPI* ModuleManager::GetModuleAPI(const std::string& name) {
    auto it = modules.find(name);
    if (it == modules.end()) {
        return nullptr;
    }
    return it->second.api;
}

void ModuleManager::StartWatching() {
    if (file_watcher && !file_watcher->IsRunning()) {
        file_watcher->Start([this](const std::string& module_name,
            const std::string& file_path,
            void* user_data) {
                this->OnFileChanged(module_name, file_path, user_data);
            });

        std::cout << "File watching started\n";
    }
}

void ModuleManager::StopWatching() {
    if (file_watcher) {
        file_watcher->Stop();
        std::cout << "File watching stopped\n";
    }
}

void ModuleManager::OnFileChanged(const std::string& module_name,
    const std::string& file_path,
    void* user_data) {
    std::cout << "Detected change in: " << file_path << "\n";
    ReloadModule(module_name);
}

void ModuleManager::UpdateModules(float delta_time) {
    for (auto& [name, module] : modules) {
        if (module.api && module.api->update) {
            module.api->update(delta_time);
        }
    }
}

std::vector<std::string> ModuleManager::GetLoadedModules() const {
    std::vector<std::string> names;
    names.reserve(modules.size());
    for (const auto& [name, _] : modules) {
        names.push_back(name);
    }
    return names;
}