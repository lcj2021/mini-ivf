
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

#ifndef INCLUDE_RCF_FILESYSTEM_HPP
#define INCLUDE_RCF_FILESYSTEM_HPP

#include <cstdint>

#include <RCF/Export.hpp>
#include <RCF/StdFileSystem.hpp>

namespace RCF
{
    /// Typedef for standard C++ path type.
    typedef RCF_FILESYSTEM_NS::path Path;

    // Serialization function for RCF::Path is in FileTransferService.cpp.
    //void serialize(SF::Archive &ar, Path &p);

    RCF_EXPORT std::wstring pathToWstring(const Path & p);
    RCF_EXPORT Path wstringToPath(const std::wstring & ws);

    RCF_EXPORT Path makeCanonical(const Path& p);

    RCF_EXPORT void setLastWriteTime(const Path& p, std::uint64_t writeTime);
    RCF_EXPORT std::uint64_t getLastWriteTime(const Path& p);

    class RCF_EXPORT FsWrappers
    {
    public:
        static void copy(const RCF_FILESYSTEM_NS::path& p1, const RCF_FILESYSTEM_NS::path& p2);
        static void rename(const RCF_FILESYSTEM_NS::path& p1, const RCF_FILESYSTEM_NS::path& p2);
        static void remove(const RCF_FILESYSTEM_NS::path& p1);
        static void create_directories(const RCF_FILESYSTEM_NS::path& p1);
        static void remove_all(const RCF_FILESYSTEM_NS::path& p1);
    };

}

#endif // ! INCLUDE_RCF_FILESYSTEM_HPP

