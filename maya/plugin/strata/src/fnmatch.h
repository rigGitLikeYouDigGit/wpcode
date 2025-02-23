#pragma once

//// by alco on gisthub

#include <string>
//#include "regex.h"
#include <regex>


/*
 * Return a new string with all occurrences of 'from' replaced with 'to'
 */
std::string replace_all(const std::string& str, const char* from, const char* to)
{
    std::string result(str);
    std::string::size_type
        index = 0,
        from_len = strlen(from),
        to_len = strlen(to);
    while ((index = result.find(from, index)) != std::string::npos) {
        result.replace(index, from_len, to);
        index += to_len;
    }
    return result;
}

/*
 * Translate a shell pattern into a regular expression
 * This is a direct translation of the algorithm defined in fnmatch.py.
 */
static std::string translate(const char* pattern)
{
    int i = 0, n = strlen(pattern);
    std::string result;

    while (i < n) {
        char c = pattern[i];
        ++i;

        if (c == '*') {
            result += ".*";
        }
        else if (c == '?') {
            result += '.';
        }
        else if (c == '[') {
            int j = i;
            /*
             * The following two statements check if the sequence we stumbled
             * upon is '[]' or '[!]' because those are not valid character
             * classes.
             */
            if (j < n && pattern[j] == '!')
                ++j;
            if (j < n && pattern[j] == ']')
                ++j;
            /*
             * Look for the closing ']' right off the bat. If one is not found,
             * escape the opening '[' and continue.  If it is found, process
             * the contents of '[...]'.
             */
            while (j < n && pattern[j] != ']')
                ++j;
            if (j >= n) {
                result += "\\[";
            }
            else {
                std::string stuff = replace_all(std::string(&pattern[i], j - i), "\\", "\\\\");
                char first_char = pattern[i];
                i = j + 1;
                result += "[";
                if (first_char == '!') {
                    result += "^" + stuff.substr(1);
                }
                else if (first_char == '^') {
                    result += "\\" + stuff;
                }
                else {
                    result += stuff;
                }
                result += "]";
            }
        }
        else {
            if (isalnum(c)) {
                result += c;
            }
            else {
                result += "\\";
                result += c;
            }
        }
    }
    /*
     * Make the expression multi-line and make the dot match any character at all.
     */
    return result + "\\Z(?ms)";
}
//
//extern "C" int private_glob_match(const char* glob, const char* str)
//{
//    regex_ref regex = regex_compile(translate(glob).c_str());
//    int result = regex_matches(regex, str);
//    regex_free(regex);
//    return result;
//}

#ifdef UNITTEST_MAIN
#include <iostream>

int main(int argc, const char* argv[])
{
    struct str_test_t {
        const char* input;
        const char* result;
    } test_strings[] = {
        { "[ab]cd?efg*", "[ab]cd.efg.*\\Z(?ms)" },
        { "[iI][!^]abc[", "[iI][^^]abc\\[\\Z(?ms)" },
        { "[]abc", "\\[\\]abc\\Z(?ms)" },
        { " [!]abc", "\\ \\[\\!\\]abc\\Z(?ms)" },
        { "*g*", ".*g.*\\Z(?ms)" },
        { "[ ][^abc][!abc][*.{}][\\[\\]\\]]", "[ ][\\^abc][^abc][*.{}][\\\\[\\\\]\\\\\\]\\]\\Z(?ms)" },
        { "\\*", "\\\\.*\\Z(?ms)" },
        { "???abc", "...abc\\Z(?ms)" },
        { "[efghu", "\\[efghu\\Z(?ms)" },
    };
    for (int i = 0; i < sizeof(test_strings) / sizeof(test_strings[0]); ++i) {
        struct str_test_t s = test_strings[i];
        std::string result = translate(s.input);
        if (s.result != result) {
            std::cout << "Assertion failed: " << s.result << " != " << result << "\n";
            abort();
        }
    }
    return 0;
}
#endif