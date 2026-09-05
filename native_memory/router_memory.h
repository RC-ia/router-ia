#pragma once

#include <stdint.h>

#ifdef _WIN32
#define ROUTER_IA_API __declspec(dllexport)
#else
#define ROUTER_IA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RouterMemoryManager RouterMemoryManager;

typedef struct RouterMemoryStats {
    uint64_t h2d_calls;
    uint64_t d2h_calls;
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t sync_calls;
} RouterMemoryStats;

typedef struct RouterMemoryManagerStats {
    uint64_t h2d_calls;
    uint64_t d2h_calls;
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t sync_calls;
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t evictions;
} RouterMemoryManagerStats;

ROUTER_IA_API RouterMemoryManager* router_mem_create_ex(
    uint64_t vram_slot_bytes,
    uint32_t vram_slots,
    uint64_t ram_slot_bytes,
    uint32_t ram_slots,
    uint32_t stream_count);

ROUTER_IA_API RouterMemoryManager* router_mem_create(
    uint64_t vram_slot_bytes,
    uint32_t vram_slots,
    uint64_t ram_slot_bytes,
    uint32_t ram_slots);

ROUTER_IA_API void router_mem_destroy(RouterMemoryManager* m);
ROUTER_IA_API void* router_mem_vram_ptr(RouterMemoryManager* m, uint32_t slot);
ROUTER_IA_API void* router_mem_ram_ptr(RouterMemoryManager* m, uint32_t slot);
ROUTER_IA_API uint32_t router_mem_vram_slots(RouterMemoryManager* m);
ROUTER_IA_API uint32_t router_mem_ram_slots(RouterMemoryManager* m);
ROUTER_IA_API uint32_t router_mem_streams(RouterMemoryManager* m);
ROUTER_IA_API uint64_t router_mem_vram_slot_bytes(RouterMemoryManager* m);
ROUTER_IA_API uint64_t router_mem_ram_slot_bytes(RouterMemoryManager* m);
ROUTER_IA_API int router_mem_stage_host(RouterMemoryManager* m, uint32_t ram_slot, const void* src, uint64_t bytes);
ROUTER_IA_API int router_mem_h2d_async(RouterMemoryManager* m, uint32_t ram_slot, uint32_t vram_slot, uint64_t bytes);
ROUTER_IA_API int router_mem_d2h_async(RouterMemoryManager* m, uint32_t vram_slot, uint32_t ram_slot, uint64_t bytes);
ROUTER_IA_API int router_mem_sync(RouterMemoryManager* m);
ROUTER_IA_API int router_mem_zero_vram(RouterMemoryManager* m, uint32_t slot, uint64_t bytes);
ROUTER_IA_API int router_mem_copy_host_slot(RouterMemoryManager* m, uint32_t dst_slot, uint32_t src_slot, uint64_t bytes);
ROUTER_IA_API RouterMemoryStats router_mem_stats(RouterMemoryManager* m);
ROUTER_IA_API const char* router_mem_last_cuda_error(void);

// Logical block memory manager.
ROUTER_IA_API int router_mm_register_block(RouterMemoryManager* m, uint32_t block_id, const void* src, uint64_t bytes);
ROUTER_IA_API int router_mm_unregister_block(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_is_registered(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_is_resident(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_acquire(RouterMemoryManager* m, uint32_t block_id, uint64_t bytes, uint32_t* out_vram_slot);
ROUTER_IA_API int router_mm_touch(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_pin(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_unpin(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_evict(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API RouterMemoryManagerStats router_mm_stats(RouterMemoryManager* m);

// Asynchronous acquire path (Proposal #1 — added alongside the legacy
// synchronous acquire, which is left untouched).
//
// router_mm_acquire_async issues the H2D transfer on the block's managing
// stream and returns immediately WITHOUT synchronizing. The block transitions
// to the "loading" state. Callers must call router_mm_wait_acquire() before
// reading the VRAM pointer, but MAY issue several acquire_async calls (e.g.
// for the next layer's experts) while still computing on already-ready blocks.
// Eviction never reuses a slot whose owning block is loading until that
// transfer has been synchronized.
ROUTER_IA_API int router_mm_acquire_async(RouterMemoryManager* m, uint32_t block_id, uint64_t bytes, uint32_t* out_vram_slot);
ROUTER_IA_API int router_mm_wait_acquire(RouterMemoryManager* m, uint32_t block_id);
ROUTER_IA_API int router_mm_is_loading(RouterMemoryManager* m, uint32_t block_id);

#ifdef __cplusplus
}
#endif
