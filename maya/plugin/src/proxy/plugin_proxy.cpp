#include <maya/MFnPlugin.h>
#include <maya/MPxCommand.h>
#include <maya/MGlobal.h>
#include <maya/MAnimControl.h>  // For animation time
#include <maya/MTimerMessage.h>
#include <maya/MArgList.h>
#include <chrono>  // For system time
#include "module_manager.h"

// Global module manager
static ModuleManager* g_module_manager = nullptr;
static MCallbackId g_idle_callback = 0;
static std::chrono::steady_clock::time_point g_last_time;

// Idle callback for update loop
void OnIdle(float elapsedTime, float lastTime, void* clientData) {
    if (!g_module_manager) return;

    // Use C++ chrono for reliable system time
    auto current_time = std::chrono::steady_clock::now();
    auto delta = std::chrono::duration<float>(current_time - g_last_time).count();
    g_last_time = current_time;

    g_module_manager->UpdateModules(delta);
}

// Alternative: Use Maya's animation time (if you need scene time)
void OnIdleAnimTime(float elapsedTime, float lastTime, void* clientData) {
    if (!g_module_manager) return;
    
    // Get Maya's current animation time
    MTime currentTime = MAnimControl::currentTime();
    static MTime lastMayaTime = currentTime;
    
    MTime delta = currentTime - lastMayaTime;
    lastMayaTime = currentTime;
    
    g_module_manager->UpdateModules(static_cast<float>(delta.as(MTime::kSeconds)));
}

// Command to reload a specific module
class ReloadModuleCmd : public MPxCommand {
public:
    static void* creator() { return new ReloadModuleCmd(); }

    MStatus doIt(const MArgList& args) override {
        if (args.length() == 0) {
            displayError("Usage: reloadModule <module_name>");
            return MS::kFailure;
        }

        MString module_name;
        args.get(0, module_name);

        if (g_module_manager->ReloadModule(module_name.asChar())) {
            MGlobal::displayInfo("Module reloaded: " + module_name);
            return MS::kSuccess;
        }

        displayError("Failed to reload module: " + module_name);
        return MS::kFailure;
    }
};

// Command to list loaded modules
class ListModulesCmd : public MPxCommand {
public:
    static void* creator() { return new ListModulesCmd(); }

    MStatus doIt(const MArgList& args) override {
        auto modules = g_module_manager->GetLoadedModules();

        MGlobal::displayInfo("Loaded modules:");
        for (const auto& name : modules) {
            MGlobal::displayInfo("  - " + MString(name.c_str()));
        }

        return MS::kSuccess;
    }
};

// Command that calls into a module
class CallCoreCmd : public MPxCommand {
public:
    static void* creator() { return new CallCoreCmd(); }

    MStatus doIt(const MArgList& args) override {
        ModuleAPI* api = g_module_manager->GetModuleAPI("Core");
        if (!api) {
            displayError("Core module not loaded");
            return MS::kFailure;
        }

        // Call module functions...
        MGlobal::displayInfo("Called into Core module");
        return MS::kSuccess;
    }
};

MStatus initializePluginExample(MObject obj) {
    MFnPlugin plugin(obj, "YourName", "1.0", "Any");

    // Create module manager
    g_module_manager = new ModuleManager();

    // Get plugin directory
    MString plugin_path = plugin.loadPath();
    std::string base_path = plugin_path.asChar();

    // Load modules
    ModuleManager::ModuleConfig core_config;
    core_config.name = "Core";
    core_config.dll_path = base_path + "/CoreImpl.dll";
    core_config.auto_reload = true;
    //core_config.init_context = &plugin; // for illustration -
    // pass pointer to any random context data for setup
    g_module_manager->LoadModule(core_config);

    ModuleManager::ModuleConfig render_config;
    render_config.name = "Render";
    render_config.dll_path = base_path + "/RenderImpl.dll";
    render_config.auto_reload = true;
    //render_config.init_context = &plugin;
    g_module_manager->LoadModule(render_config);

    // Start watching for changes
    g_module_manager->StartWatching();

    // Register commands
    plugin.registerCommand("reloadModule", ReloadModuleCmd::creator);
    plugin.registerCommand("listModules", ListModulesCmd::creator);
    plugin.registerCommand("callCore", CallCoreCmd::creator);

    // Setup update callback - initialize time
    g_last_time = std::chrono::steady_clock::now();
    g_idle_callback = MTimerMessage::addTimerCallback(1.0f, OnIdle);

    MGlobal::displayInfo("Hot-reload plugin initialized");
    return MS::kSuccess;
}

MStatus uninitializePluginExample(MObject obj) {
    MFnPlugin plugin(obj);

    // Remove callback
    if (g_idle_callback) {
        MMessage::removeCallback(g_idle_callback);
    }

    // Cleanup modules
    if (g_module_manager) {
        g_module_manager->StopWatching();
        delete g_module_manager;
        g_module_manager = nullptr;
    }

    plugin.deregisterCommand("reloadModule");
    plugin.deregisterCommand("listModules");
    plugin.deregisterCommand("callCore");

    return MS::kSuccess;
}