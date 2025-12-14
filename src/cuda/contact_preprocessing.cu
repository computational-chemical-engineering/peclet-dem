#include "ParticleSystem.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

// -----------------------------------------------------------------------------
// Functors
// -----------------------------------------------------------------------------

// Extract PairID from Contact
struct ContactToPairID {
  __host__ __device__ unsigned long long
  operator()(const ContactConstraint &c) const {
    int idA = c.bodyA;
    int idB = c.bodyB;
    if (idB < 0)
      return (unsigned long long)
          idA; // Static/Boundary? Treat as unique "Pair" with World?
    // Canonical Pair ID: (min << 32) | max
    unsigned int u = (idA < idB) ? idA : idB;
    unsigned int v = (idA < idB) ? idB : idA;
    return ((unsigned long long)u << 32) | v;
  }
};

struct ComputeHeadFlags {
  const unsigned long long *keys;
  __host__ __device__ ComputeHeadFlags(const unsigned long long *k) : keys(k) {}
  __host__ __device__ int operator()(int idx) const {
    if (idx == 0)
      return 0;
    return (keys[idx] != keys[idx - 1]) ? 1 : 0;
  }
};

struct IsActiveContact {
  __host__ __device__ int operator()(const ContactConstraint &c) const {
    return (c.dist_current <= 0.0f) ? 1 : 0;
  }
};

// Functor to assign weights based on counts
// This is done via a scatter/gather or a second pass.
// Easier: Just use a kernel with the Run-Length limits.
// OR: Transform-Scan?
//
// Strategy:
// 1. Sort d_contacts by PairID.
// 2. Reduce-by-key to get (UniqueIDs, Counts).
// 3. We need to broadcast "Count" back to every contact.
//    Since contacts are sorted, we can use a kernel that searches the unique
//    list? No, slow. Better: inclusive_scan of counts? Standard:
//    'thrust::reduce_by_key' outputs "Unique Keys" and "Counts". We can assume
//    the output is aligned with the segments? No.
//
// Alternative: Custom Kernel "ComputeWeights" that uses the sorted nature.
// If sorted: contacts[i] and contacts[i+1] share ID?
//
// Let's implement Part 1 (Sort) first, then use a custom kernel for the
// weighting because Thrust scatter is tricky for RLE expansion without an index
// array.
//
// Actually, simple kernel:
// 1. Identify "Start of Segment" (Key[i] != Key[i-1]).
// 2. Identify "End of Segment".
//    This is tricky in parallel without a scan.
//
// Hybrid approach:
// 1. Thrust Sort.
// 2. Thrust ReduceByKey -> Get Counts per Group.
// 3. Simple Kernel: Iterate *Contacts*.
//    Find if it's a start of a group. If so, read Count corresponding to this
//    group? No, we have two arrays.
//
// Best Approach for Broadcast:
// Make an array "GroupIndex" for each contact.
// Key = PairID.
// Run "ReduceByKey" to get "Counts".
// We need to map Contact -> GroupIndex -> Count.
//
// Actually, if we just use a custom kernel after sorting:
// Kernel <<<N>>>:
//   Read Key[i].
//   Search 'left' for start, search 'right' for end? Linear search is bad if
//   N_pair=1000.
//
// Better:
// Run `reduce_by_key` to get `d_unique_keys` and `d_counts`.
// Then use `binary_search` in the kernel? No.
//
// Look at `thrust::inclusive_scan` on "IsNewSegment" flags?
// Defines `SegmentID`.
// Then `d_counts[SegmentID]` is the count.
// Procedure:
// 1. Sort.
// 2. Create `flags`: `flags[i] = (key[i] != key[i-1])`.
// 3. Scan `flags` -> `segment_ids[i]`.
// 4. Reduce-by-key (or just reduce `1`s) to get `counts_per_segment`.
// 5. `weight[i] = 1.0 / counts_per_segment[segment_ids[i]]`.
// This is rigorous and fully parallel.

struct IsDifferent {
  __host__ __device__ bool operator()(const unsigned long long &a,
                                      const unsigned long long &b) const {
    return a != b;
  }
};

__global__ void broadcast_weights_kernel(ContactConstraint *contacts,
                                         const int *segment_ids,
                                         const int *segment_counts,
                                         int num_contacts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_contacts)
    return;

  int seg_id = segment_ids[idx];
  int count = segment_counts[seg_id];

  // Write Weight
  // contacts[idx] is an object. We read-modify-write.
  // Optimization: Just write to a float* weight_ptr if we separated it?
  // Current struct has weight inside.
  ContactConstraint c = contacts[idx];
  c.weight = (count > 0) ? (1.0f / (float)count) : 1.0f;
  contacts[idx] = c;
}

// -----------------------------------------------------------------------------
// Wrapper
// -----------------------------------------------------------------------------

void sort_and_compute_contact_weights(ParticleSystemData &ps) {
  int num_contacts;
  // We need to read the count from device (or host cache).
  // Assuming ps.d_contact_count points to device memory:
  cudaMemcpy(&num_contacts, ps.d_contact_count, sizeof(int),
             cudaMemcpyDeviceToHost);

  if (num_contacts == 0)
    return;

  // Wrap Raw Pointers
  thrust::device_ptr<ContactConstraint> d_contacts_ptr(ps.d_contacts);

  // 1. Sort by PairID
  // We need to define keys. Memory overhead: N_contacts * 8 bytes.
  // We can allocate temporary storage.
  // Is d_constraint_counts large enough? N_particles * 4 bytes.
  // N_contacts can be larger than N_particles.
  // We'll use a `thrust::device_vector` for safety/ease, cached or static if
  // optimization needed later. For now: Local vector (allocation cost exists,
  // but negligible for N<100k).

  thrust::device_vector<unsigned long long> keys(num_contacts);

  // Transform Contact -> Key
  thrust::transform(d_contacts_ptr, d_contacts_ptr + num_contacts, keys.begin(),
                    ContactToPairID());

  // Sort (Code and Keys together)
  thrust::sort_by_key(keys.begin(), keys.end(), d_contacts_ptr);

  // 2. Compute Segments (SegmentID)
  // Flags: 1 if key[i] != key[i-1]
  thrust::device_vector<int> flags(num_contacts);
  thrust::device_vector<unsigned long long> shifted_keys(num_contacts);

  // Shift: keys[i-1]
  // First element is always start of segment 0.
  // adjacent_difference logic?
  // Custom transform:
  // flags[i] = (i==0) ? 0 : (keys[i] != keys[i-1]) ? 1 : 0;

  // Efficient way:
  // unique_copy count? No.

  // Use `thrust::unique_count` logic manually:
  // We need `segment_ids`: 0, 0, 0, 1, 1, 2, ...
  // This is `inclusive_scan` of the flags.

  // Generate flags
  // keys: [A, A, B, C, C]
  // diff: [1, 0, 1, 1, 0] (Wait, scan of this gives 1,1,2,3,3) -> Indices
  // 0,0,1,2,2 ? We want 0-based index. Flag[0]=0. Flag[i] = (keys[i] !=
  // keys[i-1]);

  // Let's implement 'Flag Generation' via fancy iterator or simple kernel?
  // Simple kernel is fastest to write without complex functor chains.
  // But we are in "Thrust" mode.
  // `thrust::adjacent_difference`?

  // Let's rely on `reduce_by_key` to get the counts,
  // AND `inclusive_scan` on flags to get the map.

  // a. Reduce By Key -> Get Counts (and unique keys, thrown away)
  thrust::device_vector<unsigned long long> unique_keys(num_contacts);
  thrust::device_vector<int> counts_per_segment(num_contacts);

  // Transform Iterator to extract "Active" status (1 or 0)
  auto active_iter =
      thrust::make_transform_iterator(d_contacts_ptr, IsActiveContact());

  auto end_pair =
      thrust::reduce_by_key(keys.begin(), keys.end(), active_iter,
                            unique_keys.begin(), counts_per_segment.begin());

  int num_segments = end_pair.second - counts_per_segment.begin();

  // b. Assign Segment IDs to Contacts
  // We need an array `segment_ids[num_contacts]`.
  // It can be generated by: `vector[i] = (keys[i] != keys[i-1])` then
  // `inclusive_scan`. But wait, there's a simpler way?? Actually, we don't need
  // SegmentIDs if we can just expand `counts`. `gather`? We need `segment_ids`
  // to gather.

  // Calculate Flags (Head Flags)
  thrust::device_vector<int> head_flags(num_contacts);
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_contacts),
                    head_flags.begin(),
                    ComputeHeadFlags(thrust::raw_pointer_cast(keys.data())));

  // Scan to get Segment IDs
  thrust::device_vector<int> segment_ids(num_contacts);
  thrust::inclusive_scan(head_flags.begin(), head_flags.end(),
                         segment_ids.begin());

  // 3. Broadcast Weights
  // Kernel to write weights
  int threads = 256;
  int blocks = (num_contacts + threads - 1) / threads;
  broadcast_weights_kernel<<<blocks, threads>>>(
      ps.d_contacts, thrust::raw_pointer_cast(segment_ids.data()),
      thrust::raw_pointer_cast(counts_per_segment.data()), num_contacts);

  cudaDeviceSynchronize();
}
