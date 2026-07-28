#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"
#include "rktp.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#include <omp.h>

#include <cassert>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sstream>

#define UNUSED(x) (void)(x)

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
    if (str.empty()) return result;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
            int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
            for (int i = start; i <= end; ++i) result.push_back(i);
        } else {
            result.push_back(std::strtol(token.c_str(), nullptr, 10));
        }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        // [dbg-stripped] fprintf(stderr, "RKNPU-DBG assign size=%zu\n", size);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    domain_sizes[d] += size;
                    ensure_allocator_context(d);
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            assert(false);
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    domain_sizes[i] += size;
                    ensure_allocator_context(i);
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            assert(false);
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        ensure_allocator_context(domain_id);
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required domain
    void ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) == allocator_contexts.end()) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 32; info.K = 32; info.N = 32;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            info.iommu_domain_id = domain_id;

            rknn_matmul_io_attr io_attr;
            memset(&io_attr, 0, sizeof(io_attr));
            rknn_matmul_ctx ctx = 0;
            int _mmret = rknn_matmul_create(&ctx, &info, &io_attr);
            // [dbg-stripped] fprintf(stderr, "RKNPU-DBG ensure_ctx domain=%d ret=%d ctx=%p\n", domain_id, _mmret, (void*)(uintptr_t)ctx);
            allocator_contexts[domain_id] = ctx;
        }
    }
};
static IOMMUDomainManager g_domain_manager;

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-block scaling factors for quantized weights
    std::unordered_map<size_t, std::vector<float>> quantized_tensor_scales;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<size_t, std::vector<float>> hadamard_s_vectors;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        // [dbg-stripped] fprintf(stderr, "RKNPU-DBG getalloc off=%zu size=%zu\n", tensor_offset, size);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                rknn_destroy_mem(old_ctx, it->second.mem);
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                it->second.iommu_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                it->second.mem = rknn_create_mem(new_ctx, size);
                it->second.size = size;
            }
            return it->second;
        }

        // Acquiring a domain for allocation
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);

        // Allocating a new buffer for the tensor
        TensorAllocation alloc;
        alloc.mem = rknn_create_mem(alloc_ctx, size);
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;

        GGML_ASSERT(alloc.mem != nullptr && "Failed to allocate tensor memory via RKNN API");
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_fd, offset, M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // A-matrices cache (M, K, npu_type_a, domain_id)
    std::unordered_map<std::tuple<int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    static long ctx_miss;
    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type, (int)domain_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }

        ctx_miss++;
        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        rknn_core_mask core_mask;
        switch(core_id) {
            case 0: core_mask = RKNN_NPU_CORE_0; break;
            case 1: core_mask = RKNN_NPU_CORE_1; break;
            case 2: core_mask = RKNN_NPU_CORE_2; break;
            default: core_mask = RKNN_NPU_CORE_AUTO; break;
        }

        int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};
long ggml_backend_rknpu_context::ctx_miss = 0;


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Function for acquiring a pointer for tensor data
static int _grp_dbg = 0;
static void* get_tensor_real_ptr(const struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return nullptr;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        auto* ctx = (ggml_backend_rknpu_buffer_context*)tensor->buffer->context;
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_allocs.find(offset);
        if (it != ctx->tensor_allocs.end()) {
            // [dbg-stripped] if (_grp_dbg < 60) { fprintf(stderr, "RKNPU-GRP name=%s type=%d NPU-PTR(allocs-hit) off=%zu\n", tensor->name, (int)tensor->type, offset); _grp_dbg++; }
            return it->second.mem->virt_addr;
        }
    }
    // [dbg-stripped] if (_grp_dbg < 60) { fprintf(stderr, "RKNPU-GRP name=%s type=%d data-ptr pipeline=%d\n", tensor->name, (int)tensor->type, pipeline?1:0); _grp_dbg++; }
    return tensor->data;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

// Forward decl: defined in the Buffer section below, used by MUL_MAT_ID in graph_compute.
static size_t get_tensor_slice_packed_size(const struct ggml_tensor * tensor);


// ---- phase timing (RKNPU_MM_TIME=1) ----
static bool  g_mmt_on   = getenv("RKNPU_MM_TIME") != nullptr;
static double g_mmt_ctx = 0, g_mmt_a = 0, g_mmt_c = 0, g_mmt_run = 0, g_mmt_col = 0;
static long   g_mmt_n = 0, g_mmt_rows = 0;
static inline double mmt_now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
extern "C" void rknpu_mmt_dump(const char* tag) {
    if (!g_mmt_on) return;
    fprintf(stderr, "[mmt %s] miss=%ld calls=%ld rows=%ld ctx=%.3f aprep=%.3f cbuf=%.3f run=%.3f collect=%.3f total=%.3f\n",
            tag, ggml_backend_rknpu_context::ctx_miss, g_mmt_n, g_mmt_rows, g_mmt_ctx, g_mmt_a, g_mmt_c, g_mmt_run, g_mmt_col,
            g_mmt_ctx + g_mmt_a + g_mmt_c + g_mmt_run + g_mmt_col);
    g_mmt_ctx = g_mmt_a = g_mmt_c = g_mmt_run = g_mmt_col = 0; g_mmt_n = 0; g_mmt_rows = 0; ggml_backend_rknpu_context::ctx_miss = 0;
}


// ---------------- glue-op support (RKNPU_GLUE=1) ----------------
// Cheap non-matmul ops are executed with ggml's CPU kernels but stay assigned to the
// RKNPU device, so their tensors stay in this board's buffer (critical under RPC).
// Per-output-channel weight scales (default ON). RKNPU_PERCHAN=0 -> old
// per-block behaviour (one amax broadcast across the segment's columns).
static bool rknpu_perchan_enabled() {
    static const bool on = []() {
        const char * e = getenv("RKNPU_PERCHAN");
        if (!e || !*e) return false;   // OFF by default: not yet proven better than
                                       // per-block by perplexity; opt in with =1
        return !(e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F');
    }();
    return on;
}

static bool rknpu_glue_enabled() {
    static const bool on = []() {
        const char * e = getenv("RKNPU_GLUE");
        if (!e || !*e) return false;
        // honour the value: 0 / n / f disable
        return !(e[0] == '0' || e[0] == 'n' || e[0] == 'N' || e[0] == 'f' || e[0] == 'F');
    }();
    return on;
}

static ggml_backend_dev_t rknpu_cpu_dev() {
    static ggml_backend_dev_t d = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        if (reg && ggml_backend_reg_dev_count(reg) > 0) d = ggml_backend_reg_dev_get(reg, 0);
    }
    return d;
}

static ggml_backend_t rknpu_glue_backend() {
    static ggml_backend_t be = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        ggml_backend_dev_t d = rknpu_cpu_dev();
        if (d) {
            be = ggml_backend_dev_init(d, nullptr);
            if (be) {
                // Glue ops at M=1 are tiny vector ops; a 4-thread pool here just
                // adds a barrier + contends with the main compute pool. Default 1.
                int nt = 4;  // = ggml default (original behaviour); GT=1 measured -1%
                const char * e = getenv("RKNPU_GLUE_THREADS");
                if (e && *e) nt = atoi(e);
                if (nt < 1) nt = 1;
                ggml_backend_cpu_set_n_threads(be, nt);
            }
        }
    }
    return be;
}

// A matmul B-matrix is only ours if it is a preloaded weight: a graph leaf (not the
// result of an op) and not a view. KV-cache views / runtime activations are NOT.
static bool rknpu_is_weight_src(const struct ggml_tensor * t) {
    if (!t) return false;
    if (t->view_src != nullptr) return false;
    if (t->op != GGML_OP_NONE) return false;
    return true;
}

static bool rknpu_is_glue_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE: case GGML_OP_DUP: case GGML_OP_ADD: case GGML_OP_SUB:
        case GGML_OP_MUL: case GGML_OP_DIV: case GGML_OP_SCALE: case GGML_OP_SQR:
        case GGML_OP_SQRT: case GGML_OP_RMS_NORM: case GGML_OP_NORM:
        case GGML_OP_SOFT_MAX: case GGML_OP_ROPE: case GGML_OP_GLU:
        case GGML_OP_UNARY: case GGML_OP_GET_ROWS: case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT: case GGML_OP_CLAMP: case GGML_OP_CONCAT:
        case GGML_OP_CONT: case GGML_OP_CPY: case GGML_OP_RESHAPE:
        case GGML_OP_VIEW: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
        case GGML_OP_SET_ROWS: case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_ROPE_BACK: case GGML_OP_DIAG_MASK_INF:
            return true;
        default:
            return false;
    }
}

// A src that our pipeline would requantize is stored in NPU layout -> CPU kernels
// must never read it.
static bool rknpu_has_pipeline_src(const struct ggml_tensor * op) {
    const auto& cfg = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        const struct ggml_tensor * s = op->src[i];
        if (!s) continue;
        if (cfg.resolve_op_support(s) != nullptr) return true;
    }
    return false;
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

    // Getting the current device configuration once
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    // Runs ONE [K,N] weight-slice matmul on the NPU:
    //   dst_data[m*N + Noff + n] += (A @ B) * scaleA[m] * scaleB / hadamard_div
    // Weight B lives in the DMA buffer (tensor_fd/tensor_virt_addr) at byte offset
    // weight_base_offset (+ per-segment offset). Caller resolves fd/virt/domain/scales/s_vec
    // and MUST zero dst_data beforehand. Used by MUL_MAT (base=0, whole tensor) and
    // MUL_MAT_ID (per-expert base = eid * per_expert_packed).
    auto run_slice = [&](const auto* pipeline, int tensor_fd, void* tensor_virt_addr,
                         int32_t b_domain_id, size_t weight_base_offset,
                         const std::vector<float>& scales_B_grid, const std::vector<float>& s_vec,
                         int M, int K, int N, const float* x, int x_row_stride,
                         float* dst_data) -> enum ggml_status {

        int M_op = M;
        if (M > 1) {
            M_op = rknpu2_calibration::next_power_of_two(M);
        }

        const bool is_hadamard = (pipeline->use_hadamard);
        const int K_op = is_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        const rknn_matmul_type matmul_type = pipeline->mm_type;
        const int alignment = pipeline->n_align;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        auto all_k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto all_n_segments = compute_n_segments(N, config.active_cores, alignment);

        std::vector<MatrixSegmentN> active_n_segments;
        for (const auto& seg : all_n_segments) {
            if (seg.size_n > 0) active_n_segments.push_back(seg);
        }
        if (active_n_segments.empty()) return GGML_STATUS_SUCCESS;

        const size_t num_active_segments = active_n_segments.size();
        std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs(num_active_segments);
        std::shared_ptr<rknn_tensor_mem> mem_A_shared;
        std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments(num_active_segments);

        size_t type_size_packed = 0;
        if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) type_size_packed = 2;
        else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) type_size_packed = 1;

        if (g_mmt_on) { g_mmt_n++; g_mmt_rows += M; }
        double _tt = g_mmt_on ? mmt_now() : 0.0;
        size_t current_offset_in_tensor = 0;
        for (size_t k_idx = 0; k_idx < all_k_segments.size(); ++k_idx) {
            const auto& k_seg = all_k_segments[k_idx];
            const int K_seg_op = k_seg.size_k;

            // ========== 1. Preparing contexts + bind B ==========
            for (const auto& n_seg : all_n_segments) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    if (active_n_segments[idx].offset_n == n_seg.offset_n) {
                        size_t offset_in_dma = weight_base_offset + current_offset_in_tensor;

                        matmul_ctxs[idx] = backend_ctx->get_matmul_ctx(
                            (uintptr_t)tensor_virt_addr, offset_in_dma, M_op, K_seg_op, n_seg.size_n,
                            n_seg.core_id, matmul_type, b_domain_id
                        );
                        if (!matmul_ctxs[idx] || matmul_ctxs[idx]->ctx == 0) return GGML_STATUS_FAILED;

                        auto& matmul_ctx = matmul_ctxs[idx];

                        if (!matmul_ctx->b_bound) {
                            size_t segment_size_bytes = matmul_ctx->io_attr.B.size;

                            rknn_tensor_mem* mem = rknn_create_mem_from_fd(
                                matmul_ctx->ctx,
                                tensor_fd,
                                tensor_virt_addr,
                                segment_size_bytes,
                                offset_in_dma
                            );
                            if (!mem) return GGML_STATUS_FAILED;

                            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem* m) { if (m) rknn_destroy_mem(ctx, m); };
                            matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);

                            RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");

                            matmul_ctx->b_bound = true;
                        }
                        break;
                    }
                }

                if (n_seg.size_n > 0) {
                    current_offset_in_tensor += type_size_packed > 0 ? (size_t)n_seg.size_n * K_seg_op * type_size_packed : (size_t)n_seg.size_n * K_seg_op / 2;
                }
            }

            if (g_mmt_on) { double _n = mmt_now(); g_mmt_ctx += _n - _tt; _tt = _n; }
            // ========== 2. Preparing A-matrix ==========
            std::vector<float> scales_A(M, 1.0f);
            {
                auto cache_key = std::make_tuple(M_op, K_seg_op, (int)pipeline->npu_type_a, b_domain_id);
                auto& matmul_ctx_0 = matmul_ctxs[0];

                mem_A_shared = get_tensor_buffer(backend_ctx, matmul_ctx_0->ctx, matmul_ctx_0->io_attr.A.size, cache_key, backend_ctx->a_buffer_cache);
                if (!mem_A_shared) return GGML_STATUS_FAILED;

                const int row_stride = x_row_stride;
                void* dst_base = mem_A_shared->virt_addr;

                #pragma omp parallel for
                for (int m = 0; m < M; ++m) {
                    const float* src_row = x + (size_t)m * row_stride;
                    std::vector<float> ready_row(K_seg_op);

                    if (is_hadamard) {
                        std::vector<float> signed_row(K);
                        std::vector<float> full_hadamard_row(K_op);
                        for(int k=0; k<K; ++k) signed_row[k] = src_row[k] * s_vec[k];
                        rknpu2_calibration::hadamard_transform(full_hadamard_row.data(), signed_row.data(), K, K_op);

                        memcpy(ready_row.data(), full_hadamard_row.data() + k_seg.offset_k, K_seg_op * sizeof(float));
                    } else {
                        memcpy(ready_row.data(), src_row + k_seg.offset_k, K_seg_op * sizeof(float));
                    }

                    if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                        uint16_t* dst_ptr = (uint16_t*)dst_base;
                        uint16_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu2_quantization::convert_fp32_to_fp16(ready_row.data(), dst_row, K_seg_op);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                        scales_A[m] = amax_m / 127.0f;

                        int8_t* dst_ptr = (int8_t*)dst_base;
                        int8_t* dst_row = dst_ptr + (size_t)m * K_seg_op;
                        rknpu2_quantization::quantize_fp32_to_int8(ready_row.data(), dst_row, K_seg_op, scales_A[m]);
                    }
                    else if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4) {
                        float amax_m = 0.0f;
                        for (int k = 0; k < K_seg_op; ++k) amax_m = std::max(amax_m, std::abs(ready_row[k]));
                        scales_A[m] = amax_m / 7.0f;

                        uint8_t* dst_ptr = (uint8_t*)dst_base;
                        uint8_t* dst_row = dst_ptr + (size_t)m * (K_seg_op / 2);
                        rknpu2_quantization::quantize_fp32_to_int4_packed(ready_row.data(), dst_row, K_seg_op, scales_A[m]);
                    }
                }

                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_A_shared.get(), &matmul_ctxs[idx]->io_attr.A), "set_io_mem A for core");
                }

                RKNN_CHECK(rknn_mem_sync(matmul_ctxs[0]->ctx, mem_A_shared.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
            }

            if (g_mmt_on) { double _n = mmt_now(); g_mmt_a += _n - _tt; _tt = _n; }
            // ========== 3. Preparing C-matrix ==========
            {
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    auto& matmul_ctx = matmul_ctxs[idx];
                    auto cache_key = std::make_tuple(M_op, active_n_segments[idx].size_n, active_n_segments[idx].core_id, (int)pipeline->npu_type_c, b_domain_id);

                    mem_C_segments[idx] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size, cache_key, backend_ctx->c_buffer_cache);
                    if (!mem_C_segments[idx]) return GGML_STATUS_FAILED;

                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C_segments[idx].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                }
            }

            if (g_mmt_on) { double _n = mmt_now(); g_mmt_c += _n - _tt; _tt = _n; }
            // ========== 4. Running operation ==========
            {
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    int ret = rknn_matmul_run(matmul_ctxs[idx]->ctx);
                    if (ret != RKNN_SUCC) {
                        // Handle error
                    }
                }
            }

            if (g_mmt_on) { double _n = mmt_now(); g_mmt_run += _n - _tt; _tt = _n; }
            // ========== 5. Collecting results ==========
            {
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                }

                const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;

                #pragma omp parallel for
                for (int m = 0; m < M; m++) {
                    switch (pipeline->npu_type_c) {
                        case rknpu2_configuration::NPU_TYPE_FP32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float a_over_h = scales_A[m] / hadamard_divisor;
                                const float* sB = scales_B_grid.empty() ? nullptr
                                    : scales_B_grid.data() + (size_t)k_idx * (size_t)N + (size_t)N_offset;
                                float* src_segment_base = (float*)mem_C_segments[idx]->virt_addr;
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                float* src_ptr = src_segment_base + (size_t)m * N_segment;

                                if (sB) { for(int n=0; n<N_segment; ++n) dst_ptr[n] += src_ptr[n] * (sB[n] * a_over_h); }
                                else    { for(int n=0; n<N_segment; ++n) dst_ptr[n] += src_ptr[n] * a_over_h;         }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT32: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float a_over_h = scales_A[m] / hadamard_divisor;
                                const float* sB = scales_B_grid.empty() ? nullptr
                                    : scales_B_grid.data() + (size_t)k_idx * (size_t)N + (size_t)N_offset;
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                int32_t* src_ptr = (int32_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;

                                if (sB) { for(int n=0; n<N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * (sB[n] * a_over_h); }
                                else    { for(int n=0; n<N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * a_over_h;         }
                            }
                            break;
                        }

                        case rknpu2_configuration::NPU_TYPE_INT16: {
                            for (size_t idx = 0; idx < num_active_segments; idx++) {
                                int N_offset = active_n_segments[idx].offset_n;
                                int N_segment = active_n_segments[idx].size_n;
                                const float a_over_h = scales_A[m] / hadamard_divisor;
                                const float* sB = scales_B_grid.empty() ? nullptr
                                    : scales_B_grid.data() + (size_t)k_idx * (size_t)N + (size_t)N_offset;
                                float* dst_ptr = dst_data + (size_t)m * N + N_offset;
                                int16_t* src_ptr = (int16_t*)mem_C_segments[idx]->virt_addr + (size_t)m * N_segment;

                                if (sB) { for(int n=0; n<N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * (sB[n] * a_over_h); }
                                else    { for(int n=0; n<N_segment; ++n) dst_ptr[n] += (float)src_ptr[n] * a_over_h;         }
                            }
                            break;
                        }

                        default:
                            break;
                    }
                }
            }
            if (g_mmt_on) { double _n = mmt_now(); g_mmt_col += _n - _tt; _tt = _n; }
        }
        return GGML_STATUS_SUCCESS;
    };

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];

        if (node->op == GGML_OP_MUL_MAT) {
            const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
            const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
            struct ggml_tensor* dst = node;

            const int M = (int)src1->ne[1];
            const int K = (int)src0->ne[0];
            const int N = (int)src0->ne[1];
            if (M == 0 || K == 0 || N == 0) continue;

            if (rktp::enabled() && rktp::is_split(src0)) {
                float* _tp_dst = (float*)get_tensor_real_ptr(dst);
                const float* _tp_x = (const float*)get_tensor_real_ptr(src1);
                static const bool _mdbg = getenv("RKNPU_TP_MDBG") != nullptr;
                if (_mdbg) {
                    fprintf(stderr,
                        "[rktp-M] %s M=%d K=%d N=%d | src1 t=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] cont=%d"
                        " | dst t=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] cont=%d\n",
                        src0->name ? src0->name : "?", M, K, N,
                        ggml_type_name(src1->type),
                        (long long)src1->ne[0], (long long)src1->ne[1], (long long)src1->ne[2], (long long)src1->ne[3],
                        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], (int)ggml_is_contiguous(src1),
                        ggml_type_name(dst->type),
                        (long long)dst->ne[0], (long long)dst->ne[1], (long long)dst->ne[2], (long long)dst->ne[3],
                        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3], (int)ggml_is_contiguous(dst));
                    fflush(stderr);
                }
                rktp::compute(backend, src0, _tp_x, M, _tp_dst);
                if (_mdbg) { fprintf(stderr, "[rktp-M]   ok %s\n", src0->name ? src0->name : "?"); fflush(stderr); }
                continue;
            }

            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) continue;

            ggml_backend_buffer_t src0_buffer = src0->buffer;
            auto* src0_buf_ctx = (ggml_backend_rknpu_buffer_context*)src0_buffer->context;
            size_t tensor_offset_in_virtual = (uintptr_t)src0->data - (uintptr_t)src0_buf_ctx->virtual_base;

            int32_t b_domain_id = 0;
            int tensor_fd = -1;
            void* tensor_virt_addr = nullptr;
            {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->tensor_allocs.find(tensor_offset_in_virtual);
                if (it == src0_buf_ctx->tensor_allocs.end()) {
                    // Not a requantized weight after all -> run this node with CPU kernels.
                    ggml_backend_t gb = rknpu_glue_backend();
                    if (!gb) return GGML_STATUS_FAILED;
                    struct ggml_cgraph one = ggml_graph_view(cgraph, node_i, node_i + 1);
                    enum ggml_status gst = ggml_backend_graph_compute(gb, &one);
                    if (gst != GGML_STATUS_SUCCESS) return gst;
                    continue;
                }
                tensor_fd = it->second.mem->fd;
                tensor_virt_addr = it->second.mem->virt_addr;
                b_domain_id = it->second.iommu_domain_id;
            }

            std::vector<float> s_vec;
            if (pipeline->use_hadamard) {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->hadamard_s_vectors.find(tensor_offset_in_virtual);
                GGML_ASSERT(it != src0_buf_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
                s_vec = it->second;
            }

            std::vector<float> scales_B_grid;
            if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                std::lock_guard<std::mutex> lock(src0_buf_ctx->mutex);
                auto it = src0_buf_ctx->quantized_tensor_scales.find(tensor_offset_in_virtual);
                GGML_ASSERT(it != src0_buf_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
                scales_B_grid = it->second;
            }

            float* dst_data = (float*)get_tensor_real_ptr(dst);
            memset(dst_data, 0, (size_t)M * N * sizeof(float));
            const float* x = (const float*)get_tensor_real_ptr(src1);
            int x_row_stride = (int)(src1->nb[1] / sizeof(float));

            enum ggml_status st = run_slice(pipeline, tensor_fd, tensor_virt_addr, b_domain_id, 0,
                                            scales_B_grid, s_vec, M, K, N, x, x_row_stride, dst_data);
            if (st != GGML_STATUS_SUCCESS) return st;

        } else if (node->op == GGML_OP_MUL_MAT_ID) {
            const struct ggml_tensor* as  = node->src[0]; // experts:     [K, N, n_expert]
            const struct ggml_tensor* b   = node->src[1]; // activations: [K, n_used_b, n_tokens]
            const struct ggml_tensor* ids = node->src[2]; // ids:         [n_expert_used, n_tokens] (i32)
            struct ggml_tensor* dst = node;                // out:         [N, n_expert_used, n_tokens]

            const int K       = (int)as->ne[0];
            const int N       = (int)as->ne[1];
            const int64_t n_expert = as->ne[2];
            const int n_used  = (int)ids->ne[0];
            const int n_tok   = (int)ids->ne[1];
            const int r       = (int)b->ne[1];
            if (K == 0 || N == 0 || n_used == 0 || n_expert == 0) continue;

            const auto* pipeline = config.resolve_op_support(as);
            if (!pipeline) continue;

            ggml_backend_buffer_t as_buffer = as->buffer;
            auto* as_buf_ctx = (ggml_backend_rknpu_buffer_context*)as_buffer->context;
            size_t as_off = (uintptr_t)as->data - (uintptr_t)as_buf_ctx->virtual_base;

            int32_t b_domain_id = 0;
            int tensor_fd = -1;
            void* tensor_virt_addr = nullptr;
            std::vector<float> scales_all;
            std::vector<float> s_vec;
            {
                std::lock_guard<std::mutex> lock(as_buf_ctx->mutex);
                auto it = as_buf_ctx->tensor_allocs.find(as_off);
                if (it == as_buf_ctx->tensor_allocs.end()) {
                    ggml_backend_t gb = rknpu_glue_backend();
                    if (!gb) return GGML_STATUS_FAILED;
                    struct ggml_cgraph one = ggml_graph_view(cgraph, node_i, node_i + 1);
                    enum ggml_status gst = ggml_backend_graph_compute(gb, &one);
                    if (gst != GGML_STATUS_SUCCESS) return gst;
                    continue;
                }
                tensor_fd = it->second.mem->fd;
                tensor_virt_addr = it->second.mem->virt_addr;
                b_domain_id = it->second.iommu_domain_id;

                if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                    auto sit = as_buf_ctx->quantized_tensor_scales.find(as_off);
                    GGML_ASSERT(sit != as_buf_ctx->quantized_tensor_scales.end() && "MoE scales grid not found");
                    scales_all = sit->second;
                }
                if (pipeline->use_hadamard) {
                    auto hit = as_buf_ctx->hadamard_s_vectors.find(as_off);
                    GGML_ASSERT(hit != as_buf_ctx->hadamard_s_vectors.end() && "MoE hadamard 's' vector not found");
                    s_vec = hit->second;
                }
            }

            const size_t per_expert_packed = get_tensor_slice_packed_size(as);
            const size_t blocks_per_expert = scales_all.empty() ? 0 : scales_all.size() / (size_t)n_expert;

            char* dst_base = (char*)get_tensor_real_ptr(dst);
            char* b_base   = (char*)get_tensor_real_ptr((struct ggml_tensor*)b);
            const int32_t* ids_data = (const int32_t*)get_tensor_real_ptr((struct ggml_tensor*)ids);

            // Zero the whole output (each (e,t) column written exactly once below).
            memset(dst_base, 0, (size_t)N * n_used * n_tok * sizeof(float));

            const size_t ids_stride = ids->nb[1] / sizeof(int32_t);

            // ---------- BATCHED EXPERT DISPATCH (prefill) ----------
            // Group every (token,slot) pair by expert id, then run ONE matmul per routed
            // expert with M = #routed rows. Replaces the M=1-per-(token,expert) dispatch
            // storm (~n_tok*n_used tiny NPU calls) with n_expert big ones -> compute-bound.
            // RKNPU_MOE_NOBATCH=1 -> old path. RKNPU_MOE_DBG=1 -> per-node stats.
            static const bool moe_nobatch = getenv("RKNPU_MOE_NOBATCH") != nullptr;
            static const int  moe_dbg     = getenv("RKNPU_MOE_DBG") ? atoi(getenv("RKNPU_MOE_DBG")) : 0;
            static const int  moe_mb      = getenv("RKNPU_MOE_MB") ? std::max(1, atoi(getenv("RKNPU_MOE_MB"))) : 128;

            const bool b_row_contig = (b->nb[0] == sizeof(float));

            if (!moe_nobatch && n_tok > 1 && b_row_contig) {
                std::vector<std::vector<int32_t>> jobs((size_t)n_expert);
                for (int t = 0; t < n_tok; ++t) {
                    for (int e = 0; e < n_used; ++e) {
                        int eid = ids_data[(size_t)t * ids_stride + (size_t)e];
                        if (eid < 0 || (int64_t)eid >= n_expert) continue;
                        jobs[(size_t)eid].push_back(t * n_used + e);
                    }
                }

                std::vector<float> xbuf, obuf, scales_slice;
                int n_disp = 0, m_max = 0;

                for (int64_t eid = 0; eid < n_expert; ++eid) {
                    std::vector<int32_t>& jl = jobs[(size_t)eid];
                    if (jl.empty()) continue;
                    const int Mb = (int)jl.size();
                    if (Mb > m_max) m_max = Mb;
                    n_disp++;

                    scales_slice.clear();
                    if (blocks_per_expert) {
                        scales_slice.assign(scales_all.begin() + (size_t)eid * blocks_per_expert,
                                            scales_all.begin() + ((size_t)eid + 1) * blocks_per_expert);
                    }

                    // Fixed-size chunks keep the number of distinct M_op values (and hence
                    // matmul contexts / DMA fds / A-C buffers) bounded.
                    for (int c0 = 0; c0 < Mb; c0 += moe_mb) {
                        const int Mc = std::min(moe_mb, Mb - c0);

                        // Pad to a fixed ladder so only a handful of distinct M_op values
                        // (and therefore matmul contexts / DMA fds) ever get created.
                        const int Mr = moe_mb;  // single M_op: RKNPU handle space is ~64k

                        xbuf.assign((size_t)Mr * (size_t)K, 0.0f);
                        #pragma omp parallel for
                        for (int m = 0; m < Mc; ++m) {
                            const int t = jl[c0 + m] / n_used;
                            const int e = jl[c0 + m] % n_used;
                            const float* srow = (const float*)(b_base + (size_t)(e % r) * b->nb[1]
                                                                     + (size_t)t * b->nb[2]);
                            memcpy(xbuf.data() + (size_t)m * (size_t)K, srow, (size_t)K * sizeof(float));
                        }

                        obuf.assign((size_t)Mr * (size_t)N, 0.0f);

                        enum ggml_status st = run_slice(pipeline, tensor_fd, tensor_virt_addr, b_domain_id,
                                                        (size_t)eid * per_expert_packed,
                                                        scales_slice, s_vec, Mr, K, N,
                                                        xbuf.data(), K, obuf.data());
                        if (st != GGML_STATUS_SUCCESS) return st;

                        #pragma omp parallel for
                        for (int m = 0; m < Mc; ++m) {
                            const int t = jl[c0 + m] / n_used;
                            const int e = jl[c0 + m] % n_used;
                            float* dst_col = (float*)(dst_base + (size_t)e * dst->nb[1]
                                                               + (size_t)t * dst->nb[2]);
                            memcpy(dst_col, obuf.data() + (size_t)m * (size_t)N, (size_t)N * sizeof(float));
                        }
                    }
                }

                if (moe_dbg) {
                    fprintf(stderr, "[moe-batch] K=%d N=%d n_tok=%d n_used=%d experts=%d m_max=%d\n",
                            K, N, n_tok, n_used, n_disp, m_max);
                }
            } else {
                for (int t = 0; t < n_tok; ++t) {
                    for (int e = 0; e < n_used; ++e) {
                        int eid = ids_data[(size_t)t * (ids->nb[1] / sizeof(int32_t)) + (size_t)e];
                        if (eid < 0 || (int64_t)eid >= n_expert) continue;

                        size_t base_off = (size_t)eid * per_expert_packed;

                        std::vector<float> scales_slice;
                        if (blocks_per_expert) {
                            scales_slice.assign(scales_all.begin() + (size_t)eid * blocks_per_expert,
                                                scales_all.begin() + ((size_t)eid + 1) * blocks_per_expert);
                        }

                        const float* x = (const float*)(b_base + (size_t)(e % r) * b->nb[1] + (size_t)t * b->nb[2]);
                        float* dst_col = (float*)(dst_base + (size_t)e * dst->nb[1] + (size_t)t * dst->nb[2]);

                        enum ggml_status st = run_slice(pipeline, tensor_fd, tensor_virt_addr, b_domain_id, base_off,
                                                        scales_slice, s_vec, 1, K, N, x, K, dst_col);
                        if (st != GGML_STATUS_SUCCESS) return st;
                    }
                }
            }

        } else {
            if (!rknpu_glue_enabled()) continue;

            // Batch the run of consecutive non-matmul nodes into one CPU sub-graph.
            int j = node_i;
            while (j < cgraph->n_nodes) {
                enum ggml_op o = cgraph->nodes[j]->op;
                if (o == GGML_OP_MUL_MAT || o == GGML_OP_MUL_MAT_ID) break;
                j++;
            }
            ggml_backend_t gb = rknpu_glue_backend();
            if (gb) {
                struct ggml_cgraph view = ggml_graph_view(cgraph, node_i, j);
                enum ggml_status gst = ggml_backend_graph_compute(gb, &view);
                if (gst != GGML_STATUS_SUCCESS) return gst;
            }
            node_i = j - 1;
            continue;
        }
    }

    if (g_mmt_on) rknpu_mmt_dump("graph");
    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// Packed NPU size of ONE [K,N] weight slice (ne[0]=K, ne[1]=N).
static size_t get_tensor_slice_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

// Number of expert slices stacked in a (possibly 3D) weight tensor.
// Regular 2D weights => 1. MoE expert weights (MUL_MAT_ID src0 = [K,N,n_expert]) => ne[2].
static inline int64_t get_tensor_num_experts(const struct ggml_tensor * tensor) {
    return tensor->ne[2] > 0 ? tensor->ne[2] : 1;
}

// Function for calculating a real tensor size for the NPU.
// For 3D MoE expert tensors this covers all n_expert stacked [K,N] slices.
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    if (!config.resolve_op_support(tensor)) return ggml_nbytes(tensor);
    return get_tensor_slice_packed_size(tensor) * (size_t)get_tensor_num_experts(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            rknn_destroy_mem(alloc_ctx, pair.second.mem);
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    // Initialize tensor only if it is supported by the pipeline
    if (pipeline) {
        // TP: row-split tensors live in a separate local Slice buffer, and whole-offloaded
        // tensors live entirely on the shard. Do NOT pre-allocate their full DMA here or the
        // coordinator materialises the whole model in NPU (/dev/dri) memory regardless of ship.
        if (rktp::enabled() && rktp::qualifies(tensor)) {
            return GGML_STATUS_SUCCESS;
        }
        size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
        size_t size = get_tensor_packed_size(tensor);
        ctx->get_tensor_allocation(offset, size);
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else {
        GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[(size_t)((uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base)];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float>& col_scales,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    const int ksz  = k_seg.size_k;
    const int ncol = n_seg.size_n;

    // The fp32 segment is N-major: column c occupies [c*ksz, (c+1)*ksz).
    // ksz is a multiple of k_align (>=32) so it is even -> the int4 nibble pairs
    // (2i,2i+1) never straddle a column and byte offset c*ksz/2 is exact.
    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        for (int c = 0; c < ncol; ++c) {
            rknpu2_quantization::quantize_fp32_to_int8(
                fp32_segment.data() + (size_t)c * ksz,
                (int8_t*)out_quantized.data() + (size_t)c * ksz,
                (size_t)ksz,
                col_scales[c]);
        }
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        for (int c = 0; c < ncol; ++c) {
            rknpu2_quantization::quantize_fp32_to_int4_packed(
                fp32_segment.data() + (size_t)c * ksz,
                out_quantized.data() + (size_t)c * ksz / 2,
                (size_t)ksz,
                col_scales[c]);
        }
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    // [dbg-stripped] fprintf(stderr, "RKNPU-DBG set_tensor name=%s type=%d ne0=%lld ne1=%lld off=%zu size=%zu\n", tensor->name, (int)tensor->type, (long long)tensor->ne[0], (long long)tensor->ne[1], offset, size);

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    if (pipeline && rktp::enabled() && rktp::qualifies(tensor)) {
        if (rktp::on_set_tensor(tensor, data, offset, size)) return;
    }

    if (pipeline) {
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];
        if(getenv("RKNPU_TP_DEBUG")) fprintf(stderr,"[rktp] KEEP-on-coord name=%s K=%d N=%d MB=%.1f\n", tensor->name, K, N, (double)get_tensor_packed_size(tensor)/1048576.0);
        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        // Initializing Hadamard Transform Logic
        if (pipeline->use_hadamard) {
            std::vector<float> s_vec(K_op, 1.0f);
            std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor));
            std::uniform_int_distribution<int> distrib(0, 1);

            for(int k = 0; k < K_op; ++k) {
                s_vec[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
            }

            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->hadamard_s_vectors[tensor_offset_in_virtual] = s_vec;
        }

        // Computing global scale
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        // Allocating a new buffer for a tensor
        size_t required_size = get_tensor_packed_size(tensor);
        auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
        uint8_t* tensor_dma_ptr = (uint8_t*)alloc.mem->virt_addr;

        // Computing specific hardware segments
        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        // MoE: a 3D expert tensor holds n_expert stacked [K,N] slices. Requant each slice
        // into its own region of the DMA buffer and concatenate all experts' block-scales.
        // For a normal 2D weight n_expert==1 and this reduces to the original single pass.
        const int64_t n_expert = get_tensor_num_experts(tensor);
        const size_t per_expert_packed = get_tensor_slice_packed_size(tensor);

        std::vector<float> seg_fp32;
        std::vector<uint8_t> seg_npu;
        std::vector<float> tensor_block_scales;   // concatenated across experts

        for (int64_t e = 0; e < n_expert; ++e) {
            const void* expert_data = (const char*)data + (size_t)e * tensor->nb[2];
            uint8_t* current_write_ptr = tensor_dma_ptr + offset + (size_t)e * per_expert_packed;

            // Processing individual segments block-by-block
            for (const auto& k_seg : k_segments) {
                for (const auto& n_seg : n_segments) {
                    if (n_seg.size_n == 0) continue;

                    // Dequantizing the block (from this expert's source slice)
                    dequantize_tensor_segment(seg_fp32, tensor, ctx, expert_data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);

                    // Calculating scales: one per OUTPUT CHANNEL (column) of this block.
                    // Layout is [k_seg][n] with N entries per k-segment; segments tile
                    // [0,N) contiguously so the run path indexes k_idx*N + global_n.
                    const int ksz_q  = k_seg.size_k;
                    const int ncol_q = n_seg.size_n;
                    std::vector<float> col_scales(ncol_q, 1.0f);
                    if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                        const bool  is_i4          = (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4);
                        const float quant_divisor  = is_i4 ? 7.0f : 127.0f;
                        if (rknpu_perchan_enabled()) {
                            for (int c = 0; c < ncol_q; ++c) {
                                const float * col = seg_fp32.data() + (size_t)c * ksz_q;
                                // Plain amax per channel. The entropy/KL clipping search is a
                                // fix for coarse block scales and has a fixed per-call cost
                                // (num_bins*num_steps) that would be paid once per column.
                                float amax = 0.0f;
                                for (int kk = 0; kk < ksz_q; ++kk) amax = std::max(amax, std::abs(col[kk]));
                                col_scales[c] = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                            }
                        } else {
                            // legacy: single amax for the whole block, broadcast
                            float amax = 0.0f;
                            if (is_i4) {
                                amax = rknpu2_calibration::calculate_entropy_amax(seg_fp32.data(), seg_fp32.size());
                            } else {
                                for (float val : seg_fp32) amax = std::max(amax, std::abs(val));
                            }
                            const float bs = (amax == 0.0f) ? 1.0f : amax / quant_divisor;
                            std::fill(col_scales.begin(), col_scales.end(), bs);
                        }
                    }
                    tensor_block_scales.insert(tensor_block_scales.end(), col_scales.begin(), col_scales.end());

                    // Quantizing (per column)
                    quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, col_scales, pipeline->npu_type_b);

                    // Packing into chip native layout
                    size_t bytes_written = pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);

                    current_write_ptr += bytes_written;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor_offset_in_virtual] = tensor_block_scales;
        }

        rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);
        RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");
        rktp::drop_src(data, size);   // release non-split source weight from coordinator file-mmap
    } else {
        memcpy((uint8_t*)tensor->data + offset, data, size);
        // [dbg-stripped] { static int _sd=0; if(_sd<12){ const float* fp=(const float*)data; fprintf(stderr,"RKNPU-SET plain name=%s type=%d off=%zu size=%zu d0..2= %.4f %.4f %.4f\n", tensor->name,(int)tensor->type,offset,size,(double)fp[0],(double)fp[1],(double)fp[2]); _sd++; } }
    }
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context*)buffer->context;
    size_t tensor_offset_in_virtual = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;

    std::lock_guard<std::mutex> lock(ctx->mutex);
    auto it = ctx->tensor_allocs.find(tensor_offset_in_virtual);
    if (it != ctx->tensor_allocs.end()) {
        memcpy(data, (uint8_t*)it->second.mem->virt_addr + offset, size);
        // [dbg-stripped] { static int _gt=0; if(_gt<12){ fprintf(stderr,"RKNPU-GET dma name=%s off=%zu size=%zu\n", tensor->name,offset,size); _gt++; } }
    } else {
        memcpy(data, (uint8_t*)tensor->data + offset, size);
        // [dbg-stripped] { static int _gt2=0; if(_gt2<12){ const float* fp=(const float*)((uint8_t*)tensor->data+offset); fprintf(stderr,"RKNPU-GET plain name=%s off=%zu size=%zu d0..2= %.4f %.4f %.4f\n", tensor->name,offset,size,(double)fp[0],(double)fp[1],(double)fp[2]); _gt2++; } }
    }
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::lock_guard<std::mutex> lock(ctx->mutex);

    for (auto& pair : ctx->tensor_allocs) {
        memset((uint8_t*)pair.second.mem->virt_addr, value, pair.second.size);
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    size_t full = get_tensor_packed_size(tensor);
    // TP: split weights only keep their LOCAL row-slice on this (coordinator) NPU; the
    // rest lives on the shard. Reserve only the local share so the coordinator buffer
    // isn't committed at full model size (the remote slot is never written here anyway).
    if (rktp::enabled()) {
        if (rktp::whole_ok(tensor)) {
            return 4096;   // whole tensor lives on the shard; reserve almost nothing here
        }
        if (rktp::row_ok(tensor)) {
            int N=(int)tensor->ne[1], Nloc=0, Nrem=0;
            if (rktp::split_dims(N, Nloc, Nrem) && Nloc>0 && Nloc<N) {
                size_t loc = (size_t)((double)full * (double)Nloc / (double)N);
                loc = ((loc + 4095) / 4096) * 4096;
                if (loc < full) return loc;
            }
        }
    }
    return full;
}


//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

// Presenting this device as a GPU with real memory is what lets `-sm layer` and
// rktp distribute weight layers across boards - but it also makes llama.cpp put
// the KV cache in the RKNPU buffer (no SET_ROWS), which forces callers to pass
// -nkvo and -fit off and costs a lot of decode throughput. Default to the stock
// ACCEL/0 behaviour; opt in for multi-board.
static bool rknpu_as_gpu() {
    const char * v = getenv("RKNPU_AS_GPU");
    if (v && *v && !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F')) {
        return true;
    }
    // multi-board launchers imply it
    if (getenv("RKNPU_TP_FULLNPU")) return true;
    if (getenv("RKNPU_TP_SHARD"))   return true;
    return false;
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    if (!rknpu_as_gpu()) {
        // stock behaviour: compute-only accelerator, no memory to report
        *free = 0; *total = 0;
        return;
    }
    // TP mode: report huge free so llama.cpp keeps ALL weights on RKNPU (no CPU spillover);
    // rktp then ships row-halves to the shard, so real physical stays balanced.
    if (getenv("RKNPU_TP_FULLNPU")) { *free = (size_t)128*1024*1024*1024; *total = *free; return; }
    // RK-LLAMA fix: RKNPU weights are drawn from system RAM via IOMMU domains, so
    // report real host memory. This lets -sm layer split weights across multiple
    // RKNPU devices (local + RPC) proportionally instead of dumping all on one.
    size_t mem_free = 0, mem_total = 0;
    FILE* mf = fopen("/proc/meminfo", "r");
    if (mf) {
        char line[256];
        while (fgets(line, sizeof(line), mf)) {
            unsigned long kb = 0;
            if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) mem_free = (size_t)kb * 1024;
            else if (sscanf(line, "MemTotal: %lu kB", &kb) == 1) mem_total = (size_t)kb * 1024;
        }
        fclose(mf);
    }
    // Reserve headroom for KV cache, compute buffers, mmap and coordinator overhead.
    // Headroom reserved for KV cache, compute buffers, mmap, requant temporaries + system.
    // Tunable via RKNPU_RESERVE_GB (bigger => fewer expert layers placed resident on the NPU).
    size_t reserve_gb = 3;
    if (const char* rv = getenv("RKNPU_RESERVE_GB")) { long v = atol(rv); if (v > 0) reserve_gb = (size_t)v; }
    const size_t reserve = reserve_gb * 1024 * 1024 * 1024;
    if (mem_free > reserve) mem_free -= reserve; else mem_free = 0;
    *free = mem_free;
    *total = mem_total;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    // Present as GPU only when multi-board weight distribution is requested;
    // otherwise stay ACCEL (stock) so llama.cpp does not place the KV cache here.
    return rknpu_as_gpu() ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    // Glue ops: claimed so the layer stays on this device (executed with CPU kernels).
    if (rknpu_glue_enabled() && op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        if (!rknpu_is_glue_op(op->op)) return false;
        if (rknpu_has_pipeline_src(op)) return false;
        ggml_backend_dev_t cd = rknpu_cpu_dev();
        if (!cd) return false;
        return ggml_backend_dev_supports_op(cd, op);
    }

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            if (!rknpu_is_weight_src(op->src[0])) return false;
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }

        case GGML_OP_MUL_MAT_ID: {
            // MoE experts on the NPU. Opt-in via RKNPU_MOE so the default binary is unchanged.
            if (!getenv("RKNPU_MOE")) {
                return false;
            }

            if (!rknpu_is_weight_src(op->src[0])) return false;
            const struct ggml_tensor * src0 = op->src[0]; // experts: [K, N, n_expert]
            const struct ggml_tensor * src1 = op->src[1]; // activations: [K, n_expert_used, n_tokens]
            const struct ggml_tensor * src2 = op->src[2]; // ids: [n_expert_used, n_tokens] (i32)

            if (!src0 || !src1 || !src2) {
                return false;
            }

            // Same hardware-pipeline eligibility as MUL_MAT (keys on tensor type + <2GB domain).
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // Reject zero dims
            if (src0->ne[0] == 0 || src0->ne[1] == 0 || src1->ne[0] == 0) {
                return false;
            }

            // Activation must be F32, ids must be I32
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }
            if (src2->type != GGML_TYPE_I32) {
                return false;
            }

            // K / N alignment on the per-expert [K,N] slice
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // K must match between experts and activations
            if (src1->ne[0] != src0->ne[0]) {
                return false;
            }

            // Weights and activations contiguous (ids handled directly)
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }

        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Fetch device from environment variable, default to RK3588 if not set
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device = env_device ? env_device : "RK3588";
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    ggml_backend_t _rk_bk = new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    rktp::g_be() = _rk_bk;  // capture backend for eager tensor-parallel slice build
    return _rk_bk;
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name       = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer   = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .get_alignment  = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size   = */ NULL,
        /* .get_alloc_size = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .is_host        = */ NULL,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif