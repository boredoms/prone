#ifndef LWCORESET_H_
#define LWCORESET_H_

#include "utility.hpp"

namespace LWCoreset {
coreset_t run(dataset_t const &data, size_t coreset_size);
}// namespace LWCoreset


#endif// LWCORESET_H_
