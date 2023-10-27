
//******************************************************************************
// RCF - Remote Call Framework
//
// Copyright (c) 2005 - 2023, Delta V Software. All rights reserved.
// https://www.deltavsoft.com
//
// RCF is distributed under dual licenses - closed source or GPL.
// Consult your particular license for conditions of use.
//
// If you have not purchased a commercial license, you are using RCF under GPL terms.
//
// Version: 3.4
// Contact: support <at> deltavsoft.com 
//
//******************************************************************************

/// \file

#ifndef INCLUDE_UTIL_TCHAR_HPP
#define INCLUDE_UTIL_TCHAR_HPP

#include <string>

#include <RCF/Config.hpp>
#include <RCF/Export.hpp>

#if (defined(UNICODE) || defined(_UNICODE)) && !defined(RCF_WINDOWS)
#error UNICODE and _UNICODE should only be defined for Windows builds.
#endif

namespace RCF {

    RCF_EXPORT std::wstring stringToWstring(const std::string &s);
    RCF_EXPORT std::string wstringToString(const std::wstring &ws);

#ifdef __cpp_char8_t
    RCF_EXPORT std::string u8stringToString(const std::u8string& u8s);
#else
    RCF_EXPORT std::string u8stringToString(const std::string& u8s);
#endif

#if (defined(UNICODE) || defined(_UNICODE))

    #define RCF_T(x)                            L ## x                        
    typedef std::wstring                        tstring;
    inline tstring toTstring(const std::string & s)     { return stringToWstring(s); }
    inline tstring toTstring(const std::wstring & s)    { return s; }
    inline std::string toAstring(const tstring & s)     { return wstringToString(s); }
    inline std::wstring toWstring(const tstring & s)    { return s; }

#else

    #define RCF_T(x)                            x
    typedef std::string                         tstring;
    inline tstring toTstring(const std::string & s)     { return s; }
    inline tstring toTstring(const std::wstring & ws)   { return wstringToString(ws); }
    inline std::string toAstring(const tstring & s)     { return s; }
    inline std::wstring toWstring(const tstring & s)    { return stringToWstring(s); }

#endif

} // namespace RCF

#endif // ! INCLUDE_UTIL_TCHAR_HPP
