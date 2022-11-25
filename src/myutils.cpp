#include "myutils.hpp"

template <typename T, typename U, typename V>
U convT4toU4(T inp)
{
	U ret = U();
	ret.x = (V)inp.x;
	ret.y = (V)inp.y;
	ret.w = (V)inp.w;
	ret.z = (V)inp.z;
	return ret;
}

template <typename T, typename U>
T multT4byU(T type4, U typeU)
{
	type4.x *= typeU;
	type4.y *= typeU;
	type4.z *= typeU;
	type4.w *= typeU;
	return type4;
}

template <typename T>
T sumT4(T a, T b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}
