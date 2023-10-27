
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

#include <RCF/FileSystem.hpp>

#include <RCF/BsdSockets.hpp>
#include <RCF/ErrorMsg.hpp>
#include <RCF/Tchar.hpp>

#include <sys/stat.h>
#ifdef RCF_WINDOWS
#include <sys/utime.h>
#else
#include <utime.h>
#endif

namespace RCF
{

    namespace fs = RCF_FILESYSTEM_NS;

#ifdef _MSC_VER

    std::wstring pathToWstring(const Path & p)
    {
        return p.wstring();
    }

    Path wstringToPath(const std::wstring & ws)
    {
        return Path(ws);
    }

#else

    std::wstring pathToWstring(const Path & p)
    {
        std::string s = RCF::u8stringToString(p.u8string());
        std::wstring ws = RCF::stringToWstring(s);
        return ws;
    }

    Path wstringToPath(const std::wstring & ws)
    {
        std::string s = RCF::wstringToString(ws);
        return Path(s);
    }

#endif

    void FsWrappers::copy(const RCF_FILESYSTEM_NS::path& p1, const RCF_FILESYSTEM_NS::path& p2)
    {
        try
        {
            fs::copy(p1, p2);
        }
        catch ( const std::exception& e )
        {
            RCF_THROW( RCF::Exception(RcfError_UnableToCopyFile, p1.generic_u8string(), p2.generic_u8string(), e.what()) );
        }
    }

    void FsWrappers::rename(const RCF_FILESYSTEM_NS::path& p1, const RCF_FILESYSTEM_NS::path& p2)
    {
        try
        {
            fs::rename(p1, p2);
        }
        catch ( const std::exception& e )
        {
            RCF_THROW( RCF::Exception(RcfError_UnableToRenameFile, p1.generic_u8string(), p2.generic_u8string(), e.what()) );
        }
    }

    void FsWrappers::remove(const RCF_FILESYSTEM_NS::path& p1)
    {
        try
        {
            fs::remove(p1);
        }
        catch ( const std::exception& e )
        {
            RCF_THROW( RCF::Exception(RcfError_UnableToRemoveFile, p1.generic_u8string(), e.what()) );
        }
    }

    void FsWrappers::create_directories(const RCF_FILESYSTEM_NS::path& p1)
    {
        try
        {
            fs::create_directories(p1);
        }
        catch ( const std::exception& e )
        {
            RCF_THROW( RCF::Exception(RcfError_UnableToCreateDir, p1.generic_u8string(), e.what()) );
        }
    }

    void FsWrappers::remove_all(const RCF_FILESYSTEM_NS::path& p1)
    {
        try
        {
            fs::remove_all(p1);
        }
        catch ( const std::exception& e )
        {
            RCF_THROW( RCF::Exception(RcfError_UnableToRemoveDir, p1.generic_u8string(), e.what()) );
        }
    }

    Path makeCanonical(const Path& p)
    {
        //return fs::canonical(p);

        bool isUncPath = false;
        if ( p.u8string().substr(0, 2) == u8"//" )
        {
            isUncPath = true;
        }

        Path abs_p = p;

        Path result;
        for ( Path::iterator it = abs_p.begin();
        it != abs_p.end();
            ++it )
        {
            if ( *it == ".." )
            {
                // /a/b/.. is not necessarily /a if b is a symbolic link
                if ( fs::is_symlink(result) )
                {
                    result /= *it;
                }
                // /a/b/../.. is not /a/b/.. under most circumstances
                // We can end up with ..s in our result because of symbolic links
                else if ( result.filename() == ".." )
                {
                    result /= *it;
                }
                // Otherwise it should be safe to resolve the parent
                else
                {
                    result = result.parent_path();
                }
            }
            else if ( *it == "." )
            {
                // Ignore
            }
            else
            {
                // Just cat other path entries
                result /= *it;
            }
        }

        // Code above collapses the leading double slash for a UNC path. So here we put it back in.
        if ( isUncPath )
        {
            result = Path(*result.begin()) / result;
        }

        return result;
    }

    // C++ standard is very hazy on file modification times, so we just use the old C functions.

#ifdef RCF_WINDOWS

#define RCF_st      _stati64
#define RCF_stat    _wstati64
#define RCF_ut      _utimbuf
#define RCF_utime   _wutime

#else

// Unix build
#define RCF_st      stat64
#define RCF_stat    stat64
#define RCF_ut      utimbuf
#define RCF_utime   utime

#endif

#define RCF_QUOTE(x) #x
#define RCF_STR(x) RCF_QUOTE(x)


    void setLastWriteTime(const Path& p, std::uint64_t writeTime)
    {
        struct RCF_st buf = { 0 };

#ifdef RCF_WINDOWS
        // Use UTF-16 for paths, even if we are compiling in non-Unicode.
        int ret = RCF_stat(p.wstring().c_str(), &buf);
#else
        // Use UTF-8.
        int ret = RCF_stat(p.u8string().c_str(), &buf);
#endif

        if (ret != 0)
        {
            int err = Platform::OS::BsdSockets::GetLastError();
            RCF_THROW( RCF::Exception(RcfError_SetFileModTime, p.u8string(), RCF_STR(RCF_stat) "()", ret, Platform::OS::GetErrorString(err)));
        }

        std::uint64_t createTime = buf.st_atime;

        struct RCF_ut ut = { 0 };
        ut.actime = createTime;
        ut.modtime = writeTime;

#ifdef RCF_WINDOWS
        ret = RCF_utime(p.wstring().c_str(), &ut);
#else
        ret = RCF_utime(p.u8string().c_str(), &ut);
#endif

        if ( ret != 0 )
        {
            int err = Platform::OS::BsdSockets::GetLastError();
            RCF_THROW( RCF::Exception(RcfError_SetFileModTime, p.u8string(), RCF_STR(RCF_utime) "()", ret, Platform::OS::GetErrorString(err)));
        }
    }

    std::uint64_t getLastWriteTime(const Path& p)
    {
        struct RCF_st buf = { 0 };

#ifdef RCF_WINDOWS
        // Use UTF-16 for paths, even if we are compiling in non-Unicode.
        int ret = RCF_stat(p.wstring().c_str(), &buf);
#else
        // Use UTF-8.
        int ret = RCF_stat(p.u8string().c_str(), &buf);
#endif
        
        if (ret != 0)
        {
            int err = Platform::OS::BsdSockets::GetLastError();
            throw RCF::Exception(RcfError_GetFileModTime, p.u8string(), Platform::OS::GetErrorString(err));
        }

        std::uint64_t writeTime = (std::uint64_t)buf.st_mtime;
        if ( writeTime == std::uint64_t(-1) )
        {
            // Can happen on Windows if the file has a file-modified timestamp before 1970.
            writeTime = 0;
        }
        
        return writeTime;
    }
    
}
