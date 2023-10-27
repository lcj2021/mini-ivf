
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

#ifndef INCLUDE_RCF_CURRENTSESSION_HPP
#define INCLUDE_RCF_CURRENTSESSION_HPP

#include <memory>

namespace RCF {

    class RcfSession;
    typedef std::shared_ptr<RcfSession> RcfSessionPtr;

    class CurrentRcfSessionSentry
    {
    public:
        CurrentRcfSessionSentry(RcfSession & session);
        CurrentRcfSessionSentry(RcfSessionPtr sessionPtr);
        ~CurrentRcfSessionSentry();        
    };

} // namespace RCF

#endif // ! INCLUDE_RCF_CURRENTSESSION_HPP
