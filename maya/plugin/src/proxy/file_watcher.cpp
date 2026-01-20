#include "file_watcher.h"
#include <algorithm>
#include <shlwapi.h>

#pragma comment(lib, "shlwapi.lib")

FileWatcher::FileWatcher()
    : running(false)
    , should_stop(false)
    , change_event(CreateEvent(NULL, FALSE, FALSE, NULL))
{
}

FileWatcher::~FileWatcher() {
    Stop();
    if (change_event) {
        CloseHandle(change_event);
    }
}

void FileWatcher::AddWatch(const WatchConfig& config) {
    WatchEntry entry;
    entry.config = config;

    // Split path into directory and filename
    char full_path[MAX_PATH];
    GetFullPathNameA(config.file_path.c_str(), MAX_PATH, full_path, NULL);
    entry.config.file_path = full_path;

    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char fname[_MAX_FNAME];
    char ext[_MAX_EXT];
    _splitpath_s(full_path, drive, dir, fname, ext);

    entry.directory = std::string(drive) + std::string(dir);
    entry.filename = std::string(fname) + std::string(ext);

    // Get initial timestamp
    GetFileWriteTime(entry.config.file_path, entry.last_write_time);

    watches.push_back(entry);

    // Signal thread to pick up new watch
    if (change_event) {
        SetEvent(change_event);
    }
}

void FileWatcher::RemoveWatch(const std::string& module_name) {
    watches.erase(
        std::remove_if(watches.begin(), watches.end(),
            [&](const WatchEntry& e) {
                return e.config.module_name == module_name;
            }),
        watches.end()
    );
}

void FileWatcher::Start(ChangeCallback cb) {
    if (running) return;

    callback = cb;
    should_stop = false;
    running = true;

    watch_thread = std::thread(&FileWatcher::WatchThread, this);
}

void FileWatcher::Stop() {
    if (!running) return;

    should_stop = true;
    if (change_event) {
        SetEvent(change_event);
    }

    if (watch_thread.joinable()) {
        watch_thread.join();
    }

    running = false;
}

bool FileWatcher::GetFileWriteTime(const std::string& path, FILETIME& out_time) {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &data)) {
        out_time = data.ftLastWriteTime;
        return true;
    }
    return false;
}

void FileWatcher::WatchThread() {
    const DWORD POLL_INTERVAL_MS = 500;  // Check every 500ms

    // Create directory handles for change notifications
    std::vector<HANDLE> dir_handles;
    std::vector<std::string> dir_paths;

    while (!should_stop) {
        // Rebuild directory watches (in case new watches were added)
        for (auto& handle : dir_handles) {
            FindCloseChangeNotification(handle);
        }
        dir_handles.clear();
        dir_paths.clear();

        // Get unique directories to watch
        for (const auto& watch : watches) {
            if (std::find(dir_paths.begin(), dir_paths.end(), watch.directory)
                == dir_paths.end()) {
                dir_paths.push_back(watch.directory);

                HANDLE h = FindFirstChangeNotificationA(
                    watch.directory.c_str(),
                    FALSE,  // Don't watch subtree
                    FILE_NOTIFY_CHANGE_LAST_WRITE
                );

                if (h != INVALID_HANDLE_VALUE) {
                    dir_handles.push_back(h);
                }
            }
        }

        if (dir_handles.empty()) {
            // No directories to watch, just wait for signal
            WaitForSingleObject(change_event, POLL_INTERVAL_MS);
            continue;
        }

        // Add our control event to the wait list
        dir_handles.push_back(change_event);

        // Wait for any directory change or control event
        DWORD result = WaitForMultipleObjects(
            static_cast<DWORD>(dir_handles.size()),
            dir_handles.data(),
            FALSE,  // Wait for any
            POLL_INTERVAL_MS
        );

        if (should_stop) break;

        // Check all watched files for changes
        for (auto& watch : watches) {
            FILETIME current_time;
            if (GetFileWriteTime(watch.config.file_path, current_time)) {
                // Compare timestamps
                if (CompareFileTime(&current_time, &watch.last_write_time) != 0) {
                    // File changed!
                    watch.last_write_time = current_time;

                    // Small delay to ensure file is fully written
                    Sleep(100);

                    if (callback) {
                        callback(watch.config.module_name,
                            watch.config.file_path,
                            watch.config.user_data);
                    }
                }
            }
        }

        // Reset directory notifications
        for (size_t i = 0; i < dir_handles.size() - 1; ++i) {
            FindNextChangeNotification(dir_handles[i]);
        }
    }

    // Cleanup
    for (auto& handle : dir_handles) {
        if (handle != change_event) {
            FindCloseChangeNotification(handle);
        }
    }
}