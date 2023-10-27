
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

#ifndef INCLUDE_SF_QSTRINGLIST_HPP
#define INCLUDE_SF_QSTRINGLIST_HPP

#include <QStringList>

#include <SF/SerializeStl.hpp>

namespace SF {

    // QStringList
    inline void serialize_vc6(SF::Archive &ar, QStringList &t, const unsigned int)
    {
        serializeStlContainer<PushBackSemantics, NoReserveSemantics>(ar, t);
    }

} // namespace SF

#endif // ! INCLUDE_SF_QSTRINGLIST_HPP
