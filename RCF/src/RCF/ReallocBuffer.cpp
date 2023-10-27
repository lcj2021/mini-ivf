
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

#include <RCF/ReallocBuffer.hpp>

#if RCF_FEATURE_CUSTOM_ALLOCATOR==1
#include <RCF/CustomAllocator.hpp>
#endif

namespace RCF {

    ReallocBuffer::ReallocBuffer() : mpch(NULL), mSize(0), mCapacity(0)
    {
    }

    ReallocBuffer::ReallocBuffer(std::size_t size) : mpch(NULL), mSize(0), mCapacity(0)
    {
        resize(size);
    }

    void ReallocBuffer::clear()
    {
        resize(0);
    }

#if RCF_FEATURE_CUSTOM_ALLOCATOR==1

    ReallocBuffer::~ReallocBuffer()
    {
        if (mpch)
        {
            RCF_delete(mpch);
            mpch = NULL;
        }
    }

    void ReallocBuffer::resize(std::size_t newSize)
    {
        if (newSize > mCapacity)
        {
            char * mpchNew = (char *) RCF_new(newSize);
            if (mpch)
            {
                memcpy(mpchNew, mpch, mSize);
                RCF_delete(mpch);
            }
            mpch = mpchNew;
            mCapacity = newSize;

            // TODO: zero initialization for debug builds?
            // ...
        }
        mSize = newSize;
    }

#else

    ReallocBuffer::~ReallocBuffer()
    {
        if (mpch)
        {
            // realloc(..., 0) is implementation defined. On Windows, it frees the memory, on other OS's such as iOS, it doesn't.
            //mpch = (char *) realloc(mpch, 0);

            free(mpch);
            mpch = NULL;
        }
    }

    void ReallocBuffer::resize(std::size_t newSize)
    {
        if ( newSize > mCapacity )
        {
            char* pch = (char*)realloc(mpch, newSize);
            if ( !pch )
            {
                throw std::bad_alloc();
            }
            mpch = pch;
            mCapacity = newSize;

            // TODO: zero initialization for debug builds?
            // ...
        }
        mSize = newSize;
    }


#endif

    std::size_t ReallocBuffer::size()
    {
        return mSize;
    }

    std::size_t ReallocBuffer::capacity()
    {
        return mCapacity;
    }

    bool ReallocBuffer::empty()
    {
        return mSize == 0;
    }

    char * ReallocBuffer::getPtr()
    {
        return mpch;
    }

    char & ReallocBuffer::operator[](std::size_t pos)
    {
        return mpch[pos];
    }

    const char & ReallocBuffer::operator[](std::size_t pos) const
    {
        return mpch[pos];
    }

} // namespace RCF
