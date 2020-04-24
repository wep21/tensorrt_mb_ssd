#include <stdexcept>
#include <sstream>
#include <memory>
#include <type_traits>
#include <cuda_runtime_api.h>

#define CHECK_CUDA_ERROR(e) (cuda::check_error(e, __FILE__, __LINE__))

namespace cuda
{
    void check_error(const ::cudaError_t e, decltype(__FILE__) f, decltype(__LINE__) n)
	{
		if(e != ::cudaSuccess)
		{
			std::stringstream s;
			s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": " << ::cudaGetErrorString(e);
			throw std::runtime_error{s.str()};
		}
	}

	struct deleter
	{
		void operator()(void* p) const
		{
			CHECK_CUDA_ERROR(::cudaFree(p));
		}
	};
	template<typename T>
	using unique_ptr = std::unique_ptr<T, deleter>;

	// auto array = cuda::make_unique<float[]>(n);
	// ::cudaMemcpy(array.get(), src_array, sizeof(float)*n, ::cudaMemcpyHostToDevice);
	template<typename T>
	typename std::enable_if<std::is_array<T>::value, cuda::unique_ptr<T>>::type make_unique(const std::size_t n)
	{
		using U = typename std::remove_extent<T>::type;
		U* p;
		CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(U) * n));
		return cuda::unique_ptr<T>{p};
	}

	// auto value = cuda::make_unique<my_class>();
	// ::cudaMemcpy(value.get(), src_value, sizeof(my_class), ::cudaMemcpyHostToDevice);
	template<typename T>
	cuda::unique_ptr<T> make_unique()
	{
		T* p;
		CHECK_CUDA_ERROR(::cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T)));
		return cuda::unique_ptr<T>{p};
	}
}