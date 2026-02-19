#pragma once
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <vector>
#include <Windows.h>

class FileWatcher {
public:
    struct WatchConfig {
        std::string file_path;           // Absolute path to watch
        std::string module_name;         // Identifier for this module
        void* user_data;                 // Optional context
    };

    using ChangeCallback = std::function<void(const std::string& module_name,
        const std::string& file_path,
        void* user_data)>;

    FileWatcher();
    ~FileWatcher();

    // Add a file to watch
    void AddWatch(const WatchConfig& config);

    // Remove a watch by module name
    void RemoveWatch(const std::string& module_name);

    // Start watching (spawns background thread)
    void Start(ChangeCallback callback);

    // Stop watching
    void Stop();

    // Check if actively watching
    bool IsRunning() const { return running; }

private:
    struct WatchEntry {
        WatchConfig config;
        FILETIME last_write_time;
        std::string directory;
        std::string filename;
    };

    void WatchThread();
    bool GetFileWriteTime(const std::string& path, FILETIME& out_time);

    std::vector<WatchEntry> watches;
    ChangeCallback callback;
    std::thread watch_thread;
    std::atomic<bool> running;
    std::atomic<bool> should_stop;
    HANDLE change_event;  // For signaling thread to check new watches
};




