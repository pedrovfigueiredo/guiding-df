#pragma once
#include "common.h"
#include "device/cuda.h"
#include "device/atomic.h"

KRR_NAMESPACE_BEGIN

KRR_CALLABLE int32_t floatToOrderedInt(float fVal) {
#if defined(__CUDA_ARCH__)
	int32_t iVal = __float_as_int(fVal);
#else
	int32_t iVal = *reinterpret_cast<int32_t*>(&fVal);
#endif
	return (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
}

KRR_CALLABLE float orderedIntToFloat(int32_t iVal) {
	int32_t orgVal = (iVal >= 0) ? iVal : iVal ^ 0x7FFFFFFF;
#if defined(__CUDA_ARCH__)
	return __int_as_float(orgVal);
#else
	return *reinterpret_cast<float*>(&orgVal);
#endif
}


template <bool floatAsOrderedInt>
struct AtomicColor {
    using T = std::conditional_t<floatAsOrderedInt, int32_t, float>;
    atomic<T> r, g, b;

    KRR_CALLABLE AtomicColor(const Color &v) {
        if constexpr (floatAsOrderedInt) {
            r.store(floatToOrderedInt(v[0]));
            g.store(floatToOrderedInt(v[1]));
            b.store(floatToOrderedInt(v[2]));
        }
        else {
            r.store(v[0]);
            g.store(v[1]);
            b.store(v[2]);
        }
    }

    KRR_CALLABLE Color load() const {
        if constexpr (floatAsOrderedInt) {
            return Color{ orderedIntToFloat(r.load()), orderedIntToFloat(g.load()), orderedIntToFloat(b.load()) };
        }
        else {
            return Color{ r.load(), g.load(), b.load() };
        }
    }

    KRR_CALLABLE void store(const Color& c) {
        if constexpr (floatAsOrderedInt) {
            r.store(floatToOrderedInt(c[0]));
            g.store(floatToOrderedInt(c[1]));
            b.store(floatToOrderedInt(c[2]));
        }
        else {
            r.store(c[0]);
            g.store(c[1]);
            b.store(c[2]);
        }
    }

    KRR_CALLABLE void min(const Color& c) {
        if constexpr (floatAsOrderedInt) {
            r.min(floatToOrderedInt(c[0]));
            g.min(floatToOrderedInt(c[1]));
            b.min(floatToOrderedInt(c[2]));
        }
        else {
            logFatal("AtomicColor::min() not implemented for floatAsOrderedInt = false\n", true);
        }
    }

    KRR_CALLABLE void max(const Color& c) {
        if constexpr (floatAsOrderedInt) {
            r.max(floatToOrderedInt(c[0]));
            g.max(floatToOrderedInt(c[1]));
            b.max(floatToOrderedInt(c[2]));
        }
        else {
            logFatal("AtomicColor::max() not implemented for floatAsOrderedInt = false\n", true);
        }
    }

    KRR_CALLABLE void fetch_add(const Color& c) {
        if constexpr (floatAsOrderedInt) {
            r.fetch_add(floatToOrderedInt(c[0]));
            g.fetch_add(floatToOrderedInt(c[1]));
            b.fetch_add(floatToOrderedInt(c[2]));
        }
        else {
            r.fetch_add(c[0]);
            g.fetch_add(c[1]);
            b.fetch_add(c[2]);
        }
    }
};


KRR_NAMESPACE_END