
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

#include <RCF/Tchar.hpp>

#include <RCF/ThreadLocalData.hpp>

#include <locale>

namespace RCF
{

#ifndef RCF_USE_STD_UTF8_CONVERTER

    std::wstring stringToWstring(const std::string &s)
    {
        if ( s.empty() )
        {
            return std::wstring();
        }

        int wchCount = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), NULL, 0);
        RCF_VERIFY(wchCount > 0, Exception("MultiByteToWideChar() failed."));
        std::wstring ws;
        ws.resize(wchCount + 10);
        wchCount = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &ws[0], (int)ws.size());
        RCF_VERIFY(wchCount > 0, Exception("MultiByteToWideChar() failed."));
        ws.resize(wchCount);
        return ws;
    }

    std::string wstringToString(const std::wstring &ws)
    {
        if ( ws.empty() )
        {
            return std::string();
        }

        int chCount = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), NULL, 0, NULL, NULL);
        RCF_VERIFY(chCount > 0, Exception("WideCharToMultiByte() failed."));
        std::string s;
        s.resize(chCount + 10);
        chCount = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), &s[0], (int)s.size(), NULL, NULL);
        RCF_VERIFY(chCount > 0, Exception("WideCharToMultiByte() failed."));
        s.resize(chCount);
        return s;
    }

#else

    std::wstring stringToWstring(const std::string& s)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t> >& utf8Converter
            = getTlsUtf8Converter();

        return utf8Converter.from_bytes(s);
}

    std::string wstringToString(const std::wstring& ws)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t> >& utf8Converter
            = getTlsUtf8Converter();

        return utf8Converter.to_bytes(ws);
    }

#endif

#ifdef __cpp_char8_t
    std::string u8stringToString(const std::u8string& u8s)
    {
        std::string s(u8s.begin(), u8s.end());
        return s;
    }
#else
    std::string u8stringToString(const std::string& u8s)
    {
        return u8s;
    }
#endif

} // namespace RCF

