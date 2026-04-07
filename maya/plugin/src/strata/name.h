
#pragma once
#include <memory>
#include <string>
#include <cstring>
#include <string_view>

#include <algorithm>

namespace strata {

    // 32-byte string with no heap allocation
    struct StrataName {
        static constexpr size_t MAX_SIZE = 31;  // +1 for null terminator

    private:
        char data[32];  // Always stack-allocated

    public:
        // Constructors
        StrataName() { data[0] = '\0'; }

        StrataName(const char* str) {
            size_t len = std::strlen(str);
            if (len > MAX_SIZE) {
                len = MAX_SIZE;  // Truncate
            }
            std::memcpy(data, str, len);
            data[len] = '\0';
        }

        StrataName(const std::string& str) : StrataName(str.c_str()) {}

        // Access
        const char* c_str() const { return data; }
        size_t size() const { return std::strlen(data); }
        bool empty() const { return data[0] == '\0'; }

        // Conversion to std::string
        std::string str() const {
            return std::string(data);
        }

        // Implicit conversion to std::string
        operator std::string() const {
            return std::string(data);
        }

        // Comparison (for map/set)
        bool operator==(const StrataName& other) const {
            return std::strcmp(data, other.data) == 0;
        }

        bool operator<(const StrataName& other) const {
            return std::strcmp(data, other.data) < 0;
        }

        // Comparison with std::string
        bool operator==(const std::string& other) const {
            return std::strcmp(data, other.c_str()) == 0;
        }

        bool operator==(const char* other) const {
            return std::strcmp(data, other) == 0;
        }

        // String operations
        StrataName operator+(const StrataName& other) const {
            StrataName result;
            size_t my_len = size();
            size_t other_len = other.size();
            size_t total = std::min(my_len + other_len, MAX_SIZE);

            std::memcpy(result.data, data, std::min(my_len, MAX_SIZE));
            if (my_len < MAX_SIZE) {
                std::memcpy(result.data + my_len, other.data,
                    std::min(other_len, MAX_SIZE - my_len));
            }
            result.data[total] = '\0';
            return result;
        }

        // Concatenation with std::string
        std::string operator+(const std::string& other) const {
            return std::string(data) + other;
        }

        friend std::string operator+(const std::string& lhs, const StrataName& rhs) {
            return lhs + std::string(rhs.data);
        }
    };

} // namespace strata

// Hash function for unordered_map
namespace std {
    template<>
    struct hash<strata::StrataName> {
        size_t operator()(const strata::StrataName& name) const {
            // Simple hash - can use better one if needed
            return std::hash<std::string_view>{}(name.c_str());
        }
    };
}