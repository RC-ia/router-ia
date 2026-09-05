#include "router_memory.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define ROUTER_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

static constexpr uint32_t INVALID_SLOT = std::numeric_limits<uint32_t>::max();

struct MemoryBlock {
    uint32_t block_id = 0;
    uint32_t ram_slot = INVALID_SLOT;
    uint32_t vram_slot = INVALID_SLOT;
    uint64_t bytes = 0;
    uint32_t pin_count = 0;
    // Proposal #1: async-acquire state. A block whose H2D is still in flight
    // has `loading_vram_slot != INVALID_SLOT` and its `vram_slot` stays
    // INVALID until the transfer is synchronized. `loading_stream` is the
    // stream that owns the in-flight `cudaMemcpyAsync`.
    uint32_t loading_vram_slot = INVALID_SLOT;
    cudaStream_t loading_stream = nullptr;
};

struct RouterMemoryManager {
    size_t vram_slot_bytes = 0;
    size_t ram_slot_bytes = 0;
    std::vector<void*> vram_slots;
    std::vector<void*> ram_slots;
    std::vector<cudaStream_t> streams;
    RouterMemoryStats stats{};
    std::mutex mutex;

    std::unordered_map<uint32_t, MemoryBlock> blocks;
    std::vector<int32_t> vram_owner;
    std::vector<bool> ram_used;
    std::deque<uint32_t> lru_slots;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t evictions = 0;
};

static bool ok(cudaError_t e) { return e == cudaSuccess; }

static cudaStream_t stream_for_slot(RouterMemoryManager* m, uint32_t slot) {
    return m->streams[static_cast<size_t>(slot) % m->streams.size()];
}

static bool valid_bytes(const RouterMemoryManager* m, uint64_t bytes) {
    return bytes > 0 && bytes <= m->ram_slot_bytes && bytes <= m->vram_slot_bytes;
}

static void lru_remove(RouterMemoryManager* m, uint32_t slot) {
    m->lru_slots.erase(std::remove(m->lru_slots.begin(), m->lru_slots.end(), slot), m->lru_slots.end());
}

static void lru_touch(RouterMemoryManager* m, uint32_t slot) {
    lru_remove(m, slot);
    m->lru_slots.push_back(slot);
}

static bool slot_is_loading(const RouterMemoryManager* m, uint32_t vram_slot) {
    for (const auto& kv : m->blocks) {
        if (kv.second.loading_vram_slot == vram_slot) return true;
    }
    return false;
}

static int32_t find_free_vram(RouterMemoryManager* m) {
    for (uint32_t i = 0; i < m->vram_owner.size(); ++i) {
        if (m->vram_owner[i] < 0 && !slot_is_loading(m, i)) return static_cast<int32_t>(i);
    }
    return -1;
}

static int32_t find_free_ram(RouterMemoryManager* m) {
    for (uint32_t i = 0; i < m->ram_used.size(); ++i) {
        if (!m->ram_used[i]) return static_cast<int32_t>(i);
    }
    return -1;
}

static int evict_slot_locked(RouterMemoryManager* m, uint32_t vram_slot) {
    if (vram_slot >= m->vram_owner.size()) return 0;
    const int32_t owner = m->vram_owner[vram_slot];
    if (owner < 0) return 1;

    auto it = m->blocks.find(static_cast<uint32_t>(owner));
    if (it == m->blocks.end()) return 0;
    MemoryBlock& block = it->second;
    if (block.pin_count != 0) return 0;

    if (!ok(cudaMemcpyAsync(
            m->ram_slots[block.ram_slot],
            m->vram_slots[vram_slot],
            static_cast<size_t>(block.bytes),
            cudaMemcpyDeviceToHost,
            stream_for_slot(m, vram_slot)))) return 0;

    if (!ok(cudaStreamSynchronize(stream_for_slot(m, vram_slot)))) return 0;

    ++m->stats.d2h_calls;
    m->stats.bytes_d2h += block.bytes;
    ++m->stats.sync_calls;
    ++m->evictions;

    block.vram_slot = INVALID_SLOT;
    m->vram_owner[vram_slot] = -1;
    lru_remove(m, vram_slot);
    return 1;
}

ROUTER_EXPORT RouterMemoryManager* router_mem_create_ex(
    uint64_t vram_slot_bytes, uint32_t vram_slots,
    uint64_t ram_slot_bytes, uint32_t ram_slots,
    uint32_t stream_count) {
    if (vram_slot_bytes == 0 || ram_slot_bytes == 0 || vram_slots == 0 || ram_slots == 0 || stream_count == 0)
        return nullptr;

    auto* m = new RouterMemoryManager();
    m->vram_slot_bytes = static_cast<size_t>(vram_slot_bytes);
    m->ram_slot_bytes = static_cast<size_t>(ram_slot_bytes);

    m->streams.resize(stream_count, nullptr);
    for (auto& stream : m->streams) {
        if (!ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
            for (cudaStream_t s : m->streams) if (s) cudaStreamDestroy(s);
            delete m;
            return nullptr;
        }
    }

    m->vram_slots.resize(vram_slots, nullptr);
    m->ram_slots.resize(ram_slots, nullptr);
    m->vram_owner.assign(vram_slots, -1);
    m->ram_used.assign(ram_slots, false);

    for (auto& p : m->vram_slots) {
        if (!ok(cudaMalloc(&p, m->vram_slot_bytes))) {
            for (void* q : m->vram_slots) if (q) cudaFree(q);
            for (cudaStream_t s : m->streams) if (s) cudaStreamDestroy(s);
            delete m;
            return nullptr;
        }
    }

    for (auto& p : m->ram_slots) {
        if (!ok(cudaHostAlloc(&p, m->ram_slot_bytes, cudaHostAllocPortable))) {
            for (void* q : m->ram_slots) if (q) cudaFreeHost(q);
            for (void* q : m->vram_slots) if (q) cudaFree(q);
            for (cudaStream_t s : m->streams) if (s) cudaStreamDestroy(s);
            delete m;
            return nullptr;
        }
    }

    return m;
}

ROUTER_EXPORT RouterMemoryManager* router_mem_create(
    uint64_t vram_slot_bytes, uint32_t vram_slots,
    uint64_t ram_slot_bytes, uint32_t ram_slots) {
    return router_mem_create_ex(vram_slot_bytes, vram_slots, ram_slot_bytes, ram_slots, 1);
}

ROUTER_EXPORT void router_mem_destroy(RouterMemoryManager* m) {
    if (!m) return;
    for (cudaStream_t stream : m->streams) if (stream) cudaStreamSynchronize(stream);
    for (void* p : m->vram_slots) if (p) cudaFree(p);
    for (void* p : m->ram_slots) if (p) cudaFreeHost(p);
    for (cudaStream_t stream : m->streams) if (stream) cudaStreamDestroy(stream);
    delete m;
}

ROUTER_EXPORT void* router_mem_vram_ptr(RouterMemoryManager* m, uint32_t slot) {
    if (!m || slot >= m->vram_slots.size()) return nullptr;
    return m->vram_slots[slot];
}

ROUTER_EXPORT void* router_mem_ram_ptr(RouterMemoryManager* m, uint32_t slot) {
    if (!m || slot >= m->ram_slots.size()) return nullptr;
    return m->ram_slots[slot];
}

ROUTER_EXPORT uint32_t router_mem_vram_slots(RouterMemoryManager* m) {
    return m ? static_cast<uint32_t>(m->vram_slots.size()) : 0;
}

ROUTER_EXPORT uint32_t router_mem_ram_slots(RouterMemoryManager* m) {
    return m ? static_cast<uint32_t>(m->ram_slots.size()) : 0;
}

ROUTER_EXPORT uint32_t router_mem_streams(RouterMemoryManager* m) {
    return m ? static_cast<uint32_t>(m->streams.size()) : 0;
}

ROUTER_EXPORT uint64_t router_mem_vram_slot_bytes(RouterMemoryManager* m) {
    return m ? static_cast<uint64_t>(m->vram_slot_bytes) : 0;
}

ROUTER_EXPORT uint64_t router_mem_ram_slot_bytes(RouterMemoryManager* m) {
    return m ? static_cast<uint64_t>(m->ram_slot_bytes) : 0;
}

ROUTER_EXPORT int router_mem_stage_host(RouterMemoryManager* m, uint32_t ram_slot, const void* src, uint64_t bytes) {
    if (!m || !src || ram_slot >= m->ram_slots.size() || bytes > m->ram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    std::memcpy(m->ram_slots[ram_slot], src, static_cast<size_t>(bytes));
    return 1;
}

ROUTER_EXPORT int router_mem_h2d_async(RouterMemoryManager* m, uint32_t ram_slot, uint32_t vram_slot, uint64_t bytes) {
    if (!m || ram_slot >= m->ram_slots.size() || vram_slot >= m->vram_slots.size()) return 0;
    if (bytes > m->ram_slot_bytes || bytes > m->vram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    cudaError_t e = cudaMemcpyAsync(m->vram_slots[vram_slot], m->ram_slots[ram_slot], static_cast<size_t>(bytes), cudaMemcpyHostToDevice, stream_for_slot(m, vram_slot));
    if (!ok(e)) return 0;
    ++m->stats.h2d_calls;
    m->stats.bytes_h2d += bytes;
    return 1;
}

ROUTER_EXPORT int router_mem_d2h_async(RouterMemoryManager* m, uint32_t vram_slot, uint32_t ram_slot, uint64_t bytes) {
    if (!m || ram_slot >= m->ram_slots.size() || vram_slot >= m->vram_slots.size()) return 0;
    if (bytes > m->ram_slot_bytes || bytes > m->vram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    cudaError_t e = cudaMemcpyAsync(m->ram_slots[ram_slot], m->vram_slots[vram_slot], static_cast<size_t>(bytes), cudaMemcpyDeviceToHost, stream_for_slot(m, vram_slot));
    if (!ok(e)) return 0;
    ++m->stats.d2h_calls;
    m->stats.bytes_d2h += bytes;
    return 1;
}

ROUTER_EXPORT int router_mem_sync(RouterMemoryManager* m) {
    if (!m) return 0;
    for (cudaStream_t stream : m->streams) if (!ok(cudaStreamSynchronize(stream))) return 0;
    ++m->stats.sync_calls;
    return 1;
}

ROUTER_EXPORT int router_mem_zero_vram(RouterMemoryManager* m, uint32_t slot, uint64_t bytes) {
    if (!m || slot >= m->vram_slots.size() || bytes > m->vram_slot_bytes) return 0;
    return ok(cudaMemsetAsync(m->vram_slots[slot], 0, static_cast<size_t>(bytes), stream_for_slot(m, slot))) ? 1 : 0;
}

ROUTER_EXPORT int router_mem_copy_host_slot(RouterMemoryManager* m, uint32_t dst_slot, uint32_t src_slot, uint64_t bytes) {
    if (!m || dst_slot >= m->ram_slots.size() || src_slot >= m->ram_slots.size() || bytes > m->ram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    std::memcpy(m->ram_slots[dst_slot], m->ram_slots[src_slot], static_cast<size_t>(bytes));
    return 1;
}

ROUTER_EXPORT RouterMemoryStats router_mem_stats(RouterMemoryManager* m) {
    RouterMemoryStats zero{};
    if (!m) return zero;
    std::lock_guard<std::mutex> guard(m->mutex);
    return m->stats;
}

ROUTER_EXPORT const char* router_mem_last_cuda_error(void) {
    return cudaGetErrorString(cudaGetLastError());
}

ROUTER_EXPORT int router_mm_register_block(RouterMemoryManager* m, uint32_t block_id, const void* src, uint64_t bytes) {
    if (!m || !src || !valid_bytes(m, bytes)) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    if (m->blocks.find(block_id) != m->blocks.end()) return 0;

    const int32_t ram_slot = find_free_ram(m);
    if (ram_slot < 0) return 0;

    std::memcpy(m->ram_slots[ram_slot], src, static_cast<size_t>(bytes));
    m->ram_used[ram_slot] = true;

    MemoryBlock block;
    block.block_id = block_id;
    block.ram_slot = static_cast<uint32_t>(ram_slot);
    block.bytes = bytes;
    m->blocks.emplace(block_id, block);
    return 1;
}

ROUTER_EXPORT int router_mm_unregister_block(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end() || it->second.pin_count != 0) return 0;
    // If the block has an in-flight H2D (async acquire), synchronize first so
    // freeing the RAM slot cannot race the transfer reading from it.
    if (it->second.loading_vram_slot != INVALID_SLOT) {
        if (!ok(cudaStreamSynchronize(it->second.loading_stream))) return 0;
        ++m->stats.sync_calls;
    }
    if (it->second.vram_slot != INVALID_SLOT && !evict_slot_locked(m, it->second.vram_slot)) return 0;
    if (it->second.ram_slot < m->ram_used.size()) m->ram_used[it->second.ram_slot] = false;
    m->blocks.erase(it);
    return 1;
}

ROUTER_EXPORT int router_mm_is_registered(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    return m->blocks.find(block_id) != m->blocks.end() ? 1 : 0;
}

ROUTER_EXPORT int router_mm_is_resident(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    return it != m->blocks.end() && it->second.vram_slot != INVALID_SLOT ? 1 : 0;
}

ROUTER_EXPORT int router_mm_acquire(RouterMemoryManager* m, uint32_t block_id, uint64_t bytes, uint32_t* out_vram_slot) {
    if (!m || !out_vram_slot) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);

    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end()) return 0;
    MemoryBlock& block = it->second;
    if (bytes != block.bytes || !valid_bytes(m, block.bytes)) return 0;

    // If an async acquire already has this block in flight, wait for it and
    // return the now-resident slot instead of issuing a duplicate transfer.
    if (block.loading_vram_slot != INVALID_SLOT) {
        if (!ok(cudaStreamSynchronize(block.loading_stream))) return 0;
        ++m->stats.sync_calls;
        block.vram_slot = block.loading_vram_slot;
        m->vram_owner[block.vram_slot] = static_cast<int32_t>(block_id);
        block.loading_vram_slot = INVALID_SLOT;
        block.loading_stream = nullptr;
        lru_touch(m, static_cast<uint32_t>(block.vram_slot));
        *out_vram_slot = block.vram_slot;
        return 1;
    }

    if (block.vram_slot != INVALID_SLOT) {
        ++m->cache_hits;
        lru_touch(m, block.vram_slot);
        *out_vram_slot = block.vram_slot;
        return 1;
    }

    ++m->cache_misses;

    int32_t slot = find_free_vram(m);
    if (slot < 0) {
        slot = -1;
        for (uint32_t candidate : m->lru_slots) {
            const int32_t owner = m->vram_owner[candidate];
            if (owner < 0) continue;
            if (slot_is_loading(m, candidate)) continue;
            auto owner_it = m->blocks.find(static_cast<uint32_t>(owner));
            if (owner_it != m->blocks.end() && owner_it->second.pin_count == 0) {
                slot = static_cast<int32_t>(candidate);
                break;
            }
        }
        if (slot < 0) return 0;
        if (!evict_slot_locked(m, static_cast<uint32_t>(slot))) return 0;
    }

    cudaStream_t stream = stream_for_slot(m, static_cast<uint32_t>(slot));
    if (!ok(cudaMemcpyAsync(
            m->vram_slots[slot],
            m->ram_slots[block.ram_slot],
            static_cast<size_t>(block.bytes),
            cudaMemcpyHostToDevice,
            stream))) return 0;

    if (!ok(cudaStreamSynchronize(stream))) return 0;

    ++m->stats.h2d_calls;
    m->stats.bytes_h2d += block.bytes;
    ++m->stats.sync_calls;

    block.vram_slot = static_cast<uint32_t>(slot);
    m->vram_owner[slot] = static_cast<int32_t>(block_id);
    lru_touch(m, static_cast<uint32_t>(slot));
    *out_vram_slot = static_cast<uint32_t>(slot);
    return 1;
}

ROUTER_EXPORT int router_mm_touch(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end() || it->second.vram_slot == INVALID_SLOT) return 0;
    lru_touch(m, it->second.vram_slot);
    return 1;
}

ROUTER_EXPORT int router_mm_pin(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end() || it->second.vram_slot == INVALID_SLOT) return 0;
    ++it->second.pin_count;
    return 1;
}

ROUTER_EXPORT int router_mm_unpin(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end() || it->second.pin_count == 0) return 0;
    --it->second.pin_count;
    return 1;
}

ROUTER_EXPORT int router_mm_evict(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end()) return 0;
    if (it->second.vram_slot == INVALID_SLOT) return 1;
    return evict_slot_locked(m, it->second.vram_slot);
}

ROUTER_EXPORT RouterMemoryManagerStats router_mm_stats(RouterMemoryManager* m) {
    RouterMemoryManagerStats out{};
    if (!m) return out;
    std::lock_guard<std::mutex> guard(m->mutex);
    out.h2d_calls = m->stats.h2d_calls;
    out.d2h_calls = m->stats.d2h_calls;
    out.bytes_h2d = m->stats.bytes_h2d;
    out.bytes_d2h = m->stats.bytes_d2h;
    out.sync_calls = m->stats.sync_calls;
    out.cache_hits = m->cache_hits;
    out.cache_misses = m->cache_misses;
    out.evictions = m->evictions;
    return out;
}

// ---------------------------------------------------------------------------
// Proposal #1: asynchronous acquire, added alongside the legacy synchronous
// acquire. The legacy path (router_mm_acquire) is intentionally untouched.
//
// Semantics:
//  - If the block is already resident (vram_slot valid and not loading), the
//    slot is returned immediately (a cache hit, counted once).
//  - If the block is already loading, we do NOT issue a second H2D; the
//    pending transfer already targets a reserved slot, which is returned.
//  - Otherwise we reserve a free (or LRU-evictable, unlocked, non-loading)
//    slot, kick off cudaMemcpyAsync on the block's managing stream, and
//    return WITHOUT synchronizing. The block stays in the loading state.
//  - router_mm_wait_acquire(block_id) synchronizes the owning stream and
//    promotes the block to resident.
// ---------------------------------------------------------------------------

ROUTER_EXPORT int router_mm_acquire_async(RouterMemoryManager* m, uint32_t block_id, uint64_t bytes, uint32_t* out_vram_slot) {
    if (!m || !out_vram_slot) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);

    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end()) return 0;
    MemoryBlock& block = it->second;
    if (bytes != block.bytes || !valid_bytes(m, block.bytes)) return 0;

    // Already resident and not loading: plain cache hit.
    if (block.vram_slot != INVALID_SLOT) {
        ++m->cache_hits;
        lru_touch(m, block.vram_slot);
        *out_vram_slot = block.vram_slot;
        return 1;
    }

    // Already loading: reuse the reserved slot, do not double-transfer.
    if (block.loading_vram_slot != INVALID_SLOT) {
        *out_vram_slot = block.loading_vram_slot;
        return 1;
    }

    ++m->cache_misses;

    int32_t slot = find_free_vram(m);
    if (slot < 0) {
        slot = -1;
        for (uint32_t candidate : m->lru_slots) {
            const int32_t owner = m->vram_owner[candidate];
            if (owner < 0) continue;
            if (slot_is_loading(m, candidate)) continue;
            auto owner_it = m->blocks.find(static_cast<uint32_t>(owner));
            if (owner_it != m->blocks.end() && owner_it->second.pin_count == 0) {
                slot = static_cast<int32_t>(candidate);
                break;
            }
        }
        if (slot < 0) return 0;
        if (!evict_slot_locked(m, static_cast<uint32_t>(slot))) return 0;
    }

    cudaStream_t stream = stream_for_slot(m, static_cast<uint32_t>(slot));
    if (!ok(cudaMemcpyAsync(
            m->vram_slots[slot],
            m->ram_slots[block.ram_slot],
            static_cast<size_t>(block.bytes),
            cudaMemcpyHostToDevice,
            stream))) return 0;

    // Record the transfer, but DO NOT synchronize. The block stays loading
    // until router_mm_wait_acquire is called.
    ++m->stats.h2d_calls;
    m->stats.bytes_h2d += block.bytes;

    block.loading_vram_slot = static_cast<uint32_t>(slot);
    block.loading_stream = stream;
    *out_vram_slot = static_cast<uint32_t>(slot);
    return 1;
}

ROUTER_EXPORT int router_mm_is_loading(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end()) return 0;
    return it->second.loading_vram_slot != INVALID_SLOT ? 1 : 0;
}

ROUTER_EXPORT int router_mm_wait_acquire(RouterMemoryManager* m, uint32_t block_id) {
    if (!m) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    auto it = m->blocks.find(block_id);
    if (it == m->blocks.end()) return 0;
    MemoryBlock& block = it->second;

    // No in-flight transfer: nothing to wait for. If already resident, it is
    // a no-op success.
    if (block.loading_vram_slot == INVALID_SLOT) {
        return block.vram_slot != INVALID_SLOT ? 1 : 0;
    }

    if (!ok(cudaStreamSynchronize(block.loading_stream))) return 0;

    ++m->stats.sync_calls;

    // Promote loading -> resident under the reserved slot.
    block.vram_slot = block.loading_vram_slot;
    m->vram_owner[block.vram_slot] = static_cast<int32_t>(block_id);
    block.loading_vram_slot = INVALID_SLOT;
    block.loading_stream = nullptr;
    lru_touch(m, static_cast<uint32_t>(block.vram_slot));
    return 1;
}
