#pragma once

#include <string>
#include <algorithm>
#include <cctype>
#include <iterator>


/* c++ basic string processing provided by standard library challenge
* difficulty impossible
*/
namespace ed {

	inline void trimLeft(std::string str, const char* trimChars="\t\n\v\f\r ") {
		str.erase(0, str.find_first_not_of(trimChars)); // left trim
	}
	inline void trimRight(std::string str, const char* trimChars = "\t\n\v\f\r ") {
		str.erase(str.find_last_not_of(trimChars) + 1); // right trim
	}
	inline void trimEnds(std::string str, const char* trimChars = "\t\n\v\f\r ") {
		trimLeft(str);
		trimRight(str);
	}


    /*
 ***********************************************************************
 *                 C++ Wildcard Pattern Matching Library               *
 *                                                                     *
 * Author: Arash Partow (2001)                                         *
 * URL: https://www.partow.net/programming/wildcardmatching/index.html *
 *                                                                     *
 * Copyright notice:                                                   *
 * Free use of the C++ Wildcard Pattern Matching Library is permitted  *
 * under the guidelines and in accordance with the most current        *
 * version of the MIT License.                                         *
 * https://www.opensource.org/licenses/MIT                             *
 *                                                                     *
 ***********************************************************************
*/


#ifndef INCLUDE_GLOBMATCH_HPP
#define INCLUDE_GLOBMATCH_HPP


    namespace glob
    {
        namespace details
        {
            template <typename Compare,
                typename Iterator,
                typename ValueType = typename std::iterator_traits<Iterator>::value_type>
            inline bool match_impl(const Iterator pattern_begin,
                const Iterator pattern_end,
                const Iterator data_begin,
                const Iterator data_end,
                const ValueType zero_or_more,
                const ValueType exactly_one)
            {
                typedef typename std::iterator_traits<Iterator>::value_type type;

                //const Iterator null_itr(0);
                const Iterator null_itr;

                Iterator p_itr = pattern_begin;
                Iterator d_itr = data_begin;
                Iterator np_itr = null_itr;
                Iterator nd_itr = null_itr;

                for (; ; )
                {
                    if (pattern_end != p_itr)
                    {
                        const type c = *(p_itr);

                        if ((data_end != d_itr) && (Compare::cmp(c, *(d_itr)) || (exactly_one == c)))
                        {
                            ++d_itr;
                            ++p_itr;
                            continue;
                        }
                        else if (zero_or_more == c)
                        {
                            while ((pattern_end != p_itr) && (zero_or_more == *(p_itr)))
                            {
                                ++p_itr;
                            }

                            const type d = *(p_itr);

                            while ((data_end != d_itr) && !(Compare::cmp(d, *(d_itr)) || (exactly_one == d)))
                            {
                                ++d_itr;
                            }

                            // set backtrack iterators
                            np_itr = p_itr - 1;
                            nd_itr = d_itr + 1;

                            continue;
                        }
                    }
                    else if (data_end == d_itr)
                        return true;

                    if ((data_end == d_itr) || (null_itr == nd_itr))
                        return false;

                    p_itr = np_itr;
                    d_itr = nd_itr;
                }

                return true;
            }

            typedef char char_t;

            struct cs_match
            {
                static inline bool cmp(const char_t c0, const char_t c1)
                {
                    return (c0 == c1);
                }
            };

            struct cis_match
            {
                static inline bool cmp(const char_t c0, const char_t c1)
                {
                    return (std::tolower(c0) == std::tolower(c1));
                }
            };

            template <typename T>
            struct general_match
            {
                static inline bool cmp(const T v0, const T v1)
                {
                    return (v0 == v1);
                }
            };
        } // namespace details

        inline bool match(const std::string& data,
            const std::string& pattern,
            const std::string::value_type match_one_or_more = '*',
            const std::string::value_type match_exatcly_one = '?')
        {
            return details::match_impl<details::cs_match>
                (
                    std::cbegin(pattern), std::cend(pattern),
                    std::cbegin(data), std::cend(data),
                    match_one_or_more,
                    match_exatcly_one
                );
        }

        template <typename Compare>
        inline bool match(const std::string& data,
            const std::string& pattern,
            const std::string::value_type match_one_or_more = '*',
            const std::string::value_type match_exatcly_one = '?')
        {
            return details::match_impl<Compare>
                (
                    std::cbegin(pattern), std::cend(pattern),
                    std::cbegin(data), std::cend(data),
                    match_one_or_more,
                    match_exatcly_one
                );
        }
    } // namespace glob

    /* EXAMPLES ----
       const std::string data    = "How now brown cow!";
       const std::string pattern = "How*bro?n*cow?";

       if (glob::match(data, pattern))
       {
          printf("'%s' matches pattern '%s'\n",
                 data.c_str(),
                 pattern.c_str());
       }

       return 0;
    
    
    
    */

#endif

}
