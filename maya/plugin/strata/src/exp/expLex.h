#pragma once

// A simple Lexer meant to demonstrate a few theoretical concepts. It can
// support several parser concepts and is very fast (though speed is not its
// design goal).
//
// J. Arrieta, Nabla Zero Labs
//
// This code is released under the MIT License.
//
// Copyright 2018 Nabla Zero Labs
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish ,distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


/* making some adjustments to back-port this to c14, where we don't have a std string_view*/

#include <string>

#include <iomanip>
#include <iostream>

#include "../logger.h"
#include "../api.h"

//#include "span.h"

namespace strata {

    namespace expns {


        class Token {
        public:
            enum class Kind {
                Number,
                Identifier,
                LeftParen,
                RightParen,
                LeftSquare,
                RightSquare,
                LeftCurly,
                RightCurly,
                LessThan,
                GreaterThan,
                GreaterThanPipe, // projection/closest point operator
                Equal,
                Plus,
                Minus,
                Asterisk,
                Slash,
                Hash,
                Dot,
                Comma,
                Colon,
                Semicolon,
                SingleQuote,
                DoubleQuote,
                Comment,
                Pipe,
                End,
                Unexpected,
                Ampersand,
                At,
                Exclamation,
                Question,
                Tilde,
                Percent,
                Pound,
                Dollar,
                Power,
                Backslash,
                String,
                Pattern,
                Space
            };

            static inline std::string kindStrStatic(Kind val) { //// wow this language kind of sucks huh?
                switch (val) {
                case Kind::Number: return "Number";
                case Kind::Identifier: return "Identifier";
                case Kind::LeftParen: return "leftParen";
                case Kind::RightParen: return "rightParen";
                case Kind::LeftSquare: return "leftSquare";
                case Kind::RightSquare: return "rightSquare";
                case Kind::LeftCurly: return "leftCurly";
                case Kind::RightCurly: return "rightCurly";
                case Kind::LessThan: return "lessThan";
                case Kind::GreaterThan:return "greaterThan";
                case Kind::GreaterThanPipe:return "greaterThanPipe";
                case Kind::Equal: return "equal";
                case Kind::Plus: return "Plus";
                case Kind::Minus: return "minus";
                case Kind::Asterisk: return "asterisk";
                case Kind::Slash: return "slash";
                case Kind::Hash:return "hash";
                case Kind::Dot: return "dot";
                case Kind::Comma: return "comma";
                case Kind::Colon: return "colon";
                case Kind::Semicolon: return "semicolon";
                case Kind::SingleQuote: return "singleQuote";
                case Kind::DoubleQuote: return "doubleQuote";
                case Kind::Comment: return "comment";
                case Kind::Pipe: return "pipe";
                case Kind::End: return "end";
                case Kind::Unexpected: return "unexpected";
                case Kind::Ampersand: return "ampersand";
                case Kind::At: return "at";
                case Kind::Exclamation: return "exclamation";
                case Kind::Question: return "question";
                case Kind::Tilde: return "tilde";

                case Kind::Percent: return "percent";
                case Kind::Pound: return "pound";
                case Kind::Dollar: return "dollar";
                case Kind::Power: return "power";
                case Kind::Backslash: return "backslash";
                case Kind::String: return "string";
                case Kind::Pattern: return "pattern";
                case Kind::Space: return "space";
                }
                return "UNKNOWN_TOKEN";
            }

            inline std::string kindStr() {
                return kindStrStatic(getKind());
            }


            int side = 0; // -1 for left, 1 for right


            Token(Kind kind) noexcept : m_kind{ kind } {}

            Token(Kind kind, const char* beg, std::size_t len) noexcept
                : m_kind{ kind }, m_lexeme(beg, len) {}

            Token(Kind kind, const char* beg, const char* end) noexcept
                : m_kind{ kind }, m_lexeme(beg, std::distance(beg, end)) {}

            void copyOther(const Token& other) {
                m_kind = other.getKind();
                m_lexeme = other.lexeme();
            }            
            ~Token() = default;
            Token(Token const& other) {
                copyOther(other);
            }
            Token(Token&& other) = default;
            Token& operator=(Token const& other) {
                copyOther(other);
                return *this;
            }
            Token& operator=(Token&& other) = default;

            Kind getKind() const noexcept { return m_kind; }

            void getKind(Kind kind) noexcept { m_kind = kind; }

            bool is(Kind kind) const noexcept { return m_kind == kind; }

            bool is_not(Kind kind) const noexcept { return m_kind != kind; }

            bool is_one_of(Kind k1, Kind k2) const noexcept { return is(k1) || is(k2); }

            template <typename... Ts>
            bool is_one_of(Kind k1, Kind k2, Ts... ks) const noexcept {
                return is(k1) || is_one_of(k2, ks...);
            }

            //std::string_view lexeme() const noexcept { return m_lexeme; }
            std::string lexeme() const noexcept { return m_lexeme; }
            //span<const char> lexeme() const noexcept { return m_lexeme; }

            //span<const char*> 

            /*void lexeme(std::string_view lexeme) noexcept {
                m_lexeme = std::move(lexeme);
            }*/
            void lexeme(std::string lexeme) noexcept {
                m_lexeme = std::move(lexeme);
            }

            void append(std::string lex) {
                lexeme(m_lexeme + lex);
            }
            void append(Token& lex) {
                lexeme(m_lexeme + lex.lexeme());
            }

        private:
            Kind             m_kind{};
            //std::string_view m_lexeme{};
            std::string m_lexeme{};
            //span<const char> m_lexeme{};
        };

        typedef Token::Kind TKind;



        class Lexer {
        public:

            Lexer(const char* beg) noexcept : m_beg{ beg }, origChar{ beg } {}

            Lexer() {
                Lexer("a");
            }

            Token next() noexcept;

            int index() {
                return static_cast<int>(std::distance(origChar, m_beg));
            }

            void reset(const char* beg) { m_beg = beg; origChar = beg; }

            void copyOther(const Lexer& other) {
                LOG("lexer copyOther");
                m_beg = other.m_beg;
                origChar = other.origChar;
            }
            ~Lexer() = default;
            Lexer(Lexer const& other) {
                copyOther(other);
            }
            Lexer(Lexer&& other) = default;
            Lexer& operator=(Lexer const& other) {
                copyOther(other);
                return *this;
            }
            Lexer& operator=(Lexer&& other) = default;

        //private:
            Token identifier() noexcept;
            Token number() noexcept;
            Token slash_or_comment() noexcept;
            Token space() noexcept;
            Token atom(Token::Kind) noexcept;
            Token atom(Token::Kind, int) noexcept;

            char peek() const noexcept { return *m_beg; }
            char get() noexcept { return *m_beg++; }

            const char* m_beg = nullptr;
            const char* origChar = nullptr;
        };

        /*const std::string kindStr(Token::Kind kind) {
            return {} [static_cast<int>(kind)] ;
        }*/

    }
    inline std::string str(expns::Token& any) {
        return "<tk " + str(any.side) + " " + any.kindStr() + " "
            + any.lexeme() + ">";
    }
}
