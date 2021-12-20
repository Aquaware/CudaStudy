#pragma once

namespace cuda_lib
{
#if 0
}	// indent guard
#endif

#define DLL_EXPORT


#ifdef DLL_EXPORT
__declspec(dllexport) void complexCalcOriginal(int* in, int* out, int n);
__declspec(dllexport) void complexCalcFast(int* hIn, int* hOut, int n);
#else
__declspec(dllimport) void complexCalcOriginal(int* in, int* out, int n);
__declspec(dllimport) void complexCalcFast(int* in, int* out, int n);
#endif

}