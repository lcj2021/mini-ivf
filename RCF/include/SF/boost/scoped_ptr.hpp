
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

#ifndef INCLUDE_SF_SCOPED_PTR_HPP
#define INCLUDE_SF_SCOPED_PTR_HPP

namespace boost {
    template<class T>
    class scoped_ptr;
}

#include <SF/SerializeSmartPtr.hpp>

namespace SF {

    // boost::scoped_ptr
    SF_SERIALIZE_SIMPLE_SMARTPTR( boost::scoped_ptr );

}

#endif // ! INCLUDE_SF_SCOPED_PTR_HPP
