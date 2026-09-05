#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifdef _WIN32
#define ROUTER_EXPORT extern "C" __declspec(dllexport)
#else
#define ROUTER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct RouterMemStats {
    uint64_t h2d_calls;
    uint64_t d2h_calls;
    uint64_t bytes_h2d;
    uint64_t bytes_d2h;
    uint64_t sync_calls;
};

struct RouterMemory {
    size_t vram_slot_bytes = 0;
    size_t ram_slot_bytes = 0;
    std::vector<void*> vram_slots;
    std::vector<void*> ram_slots;
    std::vector<cudaStream_t> streams;
    RouterMemStats stats{};
    std::mutex mutex;
};

static bool ok(cudaError_t e) { return e == cudaSuccess; }

static cudaStream_t stream_for_slot(RouterMemory* m, uint32_t slot) {
    return m->streams[static_cast<size_t>(slot) % m->streams.size()];
}

ROUTER_EXPORT RouterMemory* router_mem_create_ex(
    uint64_t vram_slot_bytes,
    uint32_t vram_slots,
    uint64_t ram_slot_bytes,
    uint32_t ram_slots,
    uint32_t stream_count) {
    if (vram_slot_bytes == 0 || ram_slot_bytes == 0 ||
        vram_slots == 0 || ram_slots == 0 || stream_count == 0)
        return nullptr;

    auto* m = new RouterMemory();
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

ROUTER_EXPORT RouterMemory* router_mem_create(
    uint64_t vram_slot_bytes,
    uint32_t vram_slots,
    uint64_t ram_slot_bytes,
    uint32_t ram_slots) {
    return router_mem_create_ex(
        vram_slot_bytes, vram_slots, ram_slot_bytes, ram_slots, 1);
}

ROUTER_EXPORT void router_mem_destroy(RouterMemory* m) {
    if (!m) return;
    for (cudaStream_t stream : m->streams) {
        if (stream) cudaStreamSynchronize(stream);
    }
    for (void* p : m->vram_slots) if (p) cudaFree(p);
    for (void* p : m->ram_slots) if (p) cudaFreeHost(p);
    for (cudaStream_t stream : m->streams) if (stream) cudaStreamDestroy(stream);
    delete m;
}

ROUTER_EXPORT void* router_mem_vram_ptr(RouterMemory* m, uint32_t slot) {
    if (!m || slot >= m->vram_slots.size()) return nullptr;
    return m->vram_slots[slot];
}

ROUTER_EXPORT void* router_mem_ram_ptr(RouterMemory* m, uint32_t slot) {
    if (!m || slot >= m->ram_slots.size()) return nullptr;
    return m->ram_slots[slot];
}

ROUTER_EXPORT uint32_t router_mem_vram_slots(RouterMemory* m) {
    return m ? static_cast<uint32_t>(m->vram_slots.size()) : 0;
}

ROUTER_EXPORT uint32_t router_mem_ram_slots(RouterMemory* m) {
    return m ? static_cast<uint32_t>(m->ram_slots.size()) : 0;
}

ROUTER_EXPORT uint32_t router_mem_streams(RouterMemory* m) {
    return m ? static_cast<uint32_t>(m->streams.size()) : 0;
}

ROUTER_EXPORT uint64_t router_mem_vram_slot_bytes(RouterMemory* m) {
    return m ? static_cast<uint64_t>(m->vram_slot_bytes) : 0;
}

ROUTER_EXPORT uint64_t router_mem_ram_slot_bytes(RouterMemory* m) {
    return m ? static_cast<uint64_t>(m->ram_slot_bytes) : 0;
}

ROUTER_EXPORT int router_mem_stage_host(
    RouterMemory* m,
    uint32_t ram_slot,
    const void* src,
    uint64_t bytes) {
    if (!m || !src || ram_slot >= m->ram_slots.size() || bytes > m->ram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    std::memcpy(m->ram_slots[ram_slot], src, static_cast<size_t>(bytes));
    return 1;
}

ROUTER_EXPORT int router_mem_h2d_async(
    RouterMemory* m,
    uint32_t ram_slot,
    uint32_t vram_slot,
    uint64_t bytes) {
    if (!m || ram_slot >= m->ram_slots.size() || vram_slot >= m->vram_slots.size()) return 0;
    if (bytes > m->ram_slot_bytes || bytes > m->vram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    cudaError_t e = cudaMemcpyAsync(
        m->vram_slots[vram_slot],
        m->ram_slots[ram_slot],
        static_cast<size_t>(bytes),
        cudaMemcpyHostToDevice,
        stream_for_slot(m, vram_slot));
    if (!ok(e)) return 0;
    ++m->stats.h2d_calls;
    m->stats.bytes_h2d += bytes;
    return 1;
}

ROUTER_EXPORT int router_mem_d2h_async(
    RouterMemory* m,
    uint32_t vram_slot,
    uint32_t ram_slot,
    uint64_t bytes) {
    if (!m || ram_slot >= m->ram_slots.size() || vram_slot >= m->vram_slots.size()) return 0;
    if (bytes > m->ram_slot_bytes || bytes > m->vram_slot_bytes) return 0;
    std::lock_guard<std::mutex> guard(m->mutex);
    cudaError_t e = cudaMemcpyAsync(
        m->ram_slots[ram_slot],
        m->vram_slots[vram_slot],
        static_cast<size_t>(bytes),
        cudaMemcpyDeviceToHost,
        stream_for_slot(m, vram_slot));
    if (!ok(e)) return 0;
    ++m->stats.d2h_calls;
    m->stats.bytes_d2h += bytes;
    return 1;
}

ROUTER_EXPORT int router_mem_sync(RouterMemory* m) {
    if (!m) return 0;
    for (cudaStream_t stream : m->streams) {
        if (!ok(cudaStreamSynchronize(stream))) return 0;
    }
    ++m->stats.sync_calls;
    return 1;
}

ROUTER_EXPORT int router_mem_zero_vram(RouterMemory* m, uint32_t slot, uint64_t bytes) {
    if (!m || slot >= m->vram_slots.size() || bytes > m->vram_slot_bytes) return 0;
    return ok(cudaMemsetAsync(
        m->vram_slots[slot], 0, static_cast<size_t>(bytes), stream_for_slot(m, slot))) ? 1 : 0;
}

ROUTER_EXPORT int router_mem_copy_host_slot(
    RouterMemory* m,
    uint32_t dst_slot,
    uint32_t src_slot,
    uint64_t bytes) {
    if (!m || dst_slot >= m->ram_slots.size() || src_slot >= m->ram_slots.size()) return 0;
    if (bytes > m->ram_slot_bytes) return 0;
    std::memcpy(m->ram_slots[dst_slot], m->ram_slots[src_slot], static_cast<size_t>(bytes));
    return 1;
}

ROUTER_EXPORT RouterMemStats router_mem_stats(RouterMemory* m) {
    RouterMemStats zero{};
    if (!m) return zero;
    std::lock_guard<std::mutex> guard(m->mutex);
    return m->stats;
}

ROUTER_EXPORT const char* router_mem_last_cuda_error() {
    return cudaGetErrorString(cudaGetLastError());
}
