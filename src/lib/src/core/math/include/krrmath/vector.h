#pragma once

#include "common.h"
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

// template parameters are not allowed (since they contain comma) 
#define KRR_INHERIT_EIGEN(name, parent)																\
		using Eigen::parent::parent;																\
		KRR_CALLABLE name(void) : Eigen::parent() {}												\
		template <typename OtherDerived>															\
		KRR_CALLABLE name(const Eigen::MatrixBase<OtherDerived> &other) : Eigen::parent(other) {}	\
		template <typename OtherDerived>															\
		KRR_CALLABLE name &operator=(const Eigen::MatrixBase<OtherDerived> &other) {				\
			this->Eigen::parent::operator=(other);													\
			return *this;																			\
		}																					

template <typename T, int Size>
class Vector : public Eigen::Vector<T, Size> {
public:
	enum {dim = Size};

	KRR_CALLABLE Vector(void) : Eigen::Vector<T, Size>(Eigen::Vector<T, Size>::Zero()) {}

	KRR_CALLABLE Vector(T v)
		: Eigen::Vector<T, Size>(Eigen::Vector<T, Size>::Constant(v)) {}

	KRR_CALLABLE Vector(Eigen::Array<T, Size, 1> arr) : Eigen::Vector<T, Size>(arr.matrix()) {}
	
	template <typename OtherDerived>
	KRR_CALLABLE Vector(const Vector<OtherDerived, Size> &other) {
		this->Eigen::Vector<T, Size>::operator=(other.template cast<T>());
	}

	template <typename OtherDerived>
	KRR_CALLABLE Vector(const Eigen::MatrixBase<OtherDerived> &other) : 
		Eigen::Vector<T, Size>(other) {}
	
	template <typename OtherDerived>
	KRR_CALLABLE Vector &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
		this->Eigen::Vector<T, Size>::operator=(other);
		return *this;
	}

	KRR_CALLABLE bool hasInf() const { return this->array().isInf().any(); }

	KRR_CALLABLE friend Vector max(const Vector &vec1, const Vector &vec2) {
		return vec1.cwiseMax(vec2);
	}
	KRR_CALLABLE friend Vector min(const Vector &vec1, const Vector &vec2) {
		return vec1.cwiseMin(vec2);
	}
	KRR_CALLABLE friend Vector abs(const Vector &vec) { return vec.cwiseAbs(); }
	KRR_CALLABLE friend Vector sqrt(const Vector &vec) { return vec.cwiseSqrt(); }
	KRR_CALLABLE friend Vector inverse(const Vector &vec) { return vec.cwiseInverse(); }
};

template <typename T>
class Vector2 : public Vector<T, 2> {
public:
	using Vector<T, 2>::Vector;

	KRR_CALLABLE Vector2(const Vector<T, 3>& v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
	}

	KRR_CALLABLE Vector2(const T &x, const T &y) {
		this->operator[](0) = x;
		this->operator[](1) = y;
	}

#ifdef __CUDACC__

	KRR_CALLABLE operator float2() const {
		return make_float2(this->operator[](0), this->operator[](1));
	}

	KRR_CALLABLE Vector2(const float2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Vector2(const uint2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Vector2(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}
#endif

#ifdef KRR_MATH_JSON
	friend void to_json(json &j, const Vector2<T> &v) {
		for (int i = 0; i < 2; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Vector2<T> &v) {
		for (int i = 0; i < 2; i++) {
			v[i] = (T) j.at(i);
		}
	}
#endif
};

template <typename T>
class Vector3 : public Vector<T, 3> {
public:
	using Vector<T, 3>::Vector;

	KRR_CALLABLE Vector3(const Vector<T, 4>& v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
	}

	KRR_CALLABLE Vector3(const T &x, const T &y, const T &z) {
		this->operator[](0) = x;
		this->operator[](1) = y;
		this->operator[](2) = z;
	}
	
#ifdef __CUDACC__

	KRR_CALLABLE operator float3() const {
		return make_float3(this->operator[](0), 
			this->operator[](1), 
			this->operator[](2));
	}
	
	KRR_CALLABLE Vector3(const float3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}

	KRR_CALLABLE Vector3(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}
	
	KRR_CALLABLE Vector3(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}
#endif

#ifdef KRR_MATH_JSON
	friend void to_json(json &j, const Vector3<T> &v) {
		for (int i = 0; i < 3; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Vector3<T> &v) {
		for (int i = 0; i < 3; i++) {
			v[i] = (T) j.at(i);
		}
	}
#endif
};

template <typename T>
class Vector4 : public Vector<T, 4> {
public:
	using Vector<T, 4>::Vector;

	KRR_CALLABLE Vector4(const Vector3<T>& v, T w) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
		this->operator[](3) = w;
	}

	KRR_CALLABLE Vector4(const T &x, const T &y, const T &z, const T &w) {
		this->operator[](0) = x;
		this->operator[](1) = y;
		this->operator[](2) = z;
		this->operator[](3) = w;
	}

#ifdef __CUDACC__
	KRR_CALLABLE operator float4() const {
		return make_float4(this->operator[](0), this->operator[](1), this->operator[](2), this->operator[](3));
	}

	KRR_CALLABLE Vector4(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
		this->operator[](3) = v.w;
	}
#endif

#ifdef KRR_MATH_JSON
	friend void to_json(json &j, const Vector4<T> &v) {
		for (int i = 0; i < 4; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Vector4<T> &v) {
		for (int i = 0; i < 4; i++) {
			v[i] = (T) j.at(i);
		}
	}
#endif
};

using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;
using Vector2ui = Vector2<uint>;
using Vector3f = Vector3<float>;
using Vector3i = Vector3<int>;
using Vector3ui = Vector3<uint>;
using Vector4f = Vector4<float>;
using Vector4i = Vector4<int>;
using Vector4ui = Vector4<uint>;

KRR_NAMESPACE_END