// rktp.h - RKNPU tensor-parallel (row-split) client + local-slice engine.
// Split a big MUL_MAT weight by output-rows: local slice stays on this NPU,
// remote slice runs on a tp_shard daemon (another board's NPU) concurrently; concat.
// v2: eager-build local slices at load time (no host-RAM pend transient) +
//     tunable local fraction (RKNPU_TP_LOCFRAC) so the coordinator can hold less.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/mman.h>

namespace rktp {

// Release source (gguf-mmap) weight pages after requant/ship: the coordinator no
// longer needs them (remote half shipped, local half copied into NPU DMA). Frees the
// coordinator's file-mmap footprint so per-board memory balances. Page-conservative +
// env-gated (RKNPU_TP_DROPSRC). Safe on a private file mmap (re-faults from file).
inline void drop_src(const void* data, size_t size){
    static int on = -1; if(on==-1) on = getenv("RKNPU_TP_DROPSRC")?1:0;
    if(!on || !data || !size) return;
    long ps = sysconf(_SC_PAGESIZE); if(ps<=0) return;
    uintptr_t start=(uintptr_t)data, end=start+size;
    uintptr_t a=(start+ps-1)&~((uintptr_t)ps-1);   // round start UP
    uintptr_t b=end&~((uintptr_t)ps-1);            // round end DOWN (only fully-contained pages)
    if(b>a) madvise((void*)a,(size_t)(b-a),MADV_DONTNEED);
}

static const int   TP_ALIGN = 64;     // N_local aligned to this (covers n_align 16/32/64)
inline int tp_min_n(){ static int v = (getenv("RKNPU_TP_MINN")? atoi(getenv("RKNPU_TP_MINN")) : 8192); return v; }
#define TP_MIN_N tp_min_n()
// local fraction of output rows kept on THIS (coordinator) NPU; rest shipped to shard.
inline double loc_frac(){ static double v = (getenv("RKNPU_TP_LOCFRAC")? atof(getenv("RKNPU_TP_LOCFRAC")) : 0.5); if(v<=0.05||v>=0.95) v=0.5; return v; }
// fraction of the NON-row-splittable matmul weights to offload WHOLE to the shard (balance the un-splittable half).
inline double offload_frac(){ static double v = (getenv("RKNPU_TP_OFFLOAD")? atof(getenv("RKNPU_TP_OFFLOAD")) : 0.0); if(v<0) v=0; if(v>1) v=1; return v; }
// stable FNV hash of a tensor name -> [0,1000)
inline unsigned name_bucket(const char* n){ unsigned h=2166136261u; if(n) for(const char*p=n;*p;++p){h^=(unsigned char)*p;h*=16777619u;} return h%1000; }

// captured RKNPU backend (set at device_init_backend, before model load)
inline ggml_backend_t& g_be(){ static ggml_backend_t b=nullptr; return b; }

// a local weight-slice matmul on THIS NPU (persistent graph, reused)
struct Slice {
    ggml_backend_t be=nullptr; ggml_context* cw=nullptr; ggml_backend_buffer_t bw=nullptr; ggml_tensor* W=nullptr;
    int type=0,K=0,N=0,curM=-1; ggml_context* cc=nullptr; ggml_gallocr_t al=nullptr; ggml_cgraph* gf=nullptr; ggml_tensor *x=nullptr,*o=nullptr;
    void build(ggml_backend_t b,int t,int K_,int N_,const void* bytes,size_t nb);
    void run(int M,const float* xd,float* od);
};

struct Pending { int type,K,Nloc,Nrem,N; std::vector<uint8_t> loc; };  // fallback lazy path only
struct RemoteFull { int K,N,type; };  // whole tensor offloaded to shard (computed there)

inline bool& in_build(){ static thread_local bool v=false; return v; }
inline std::mutex& mx(){ static std::mutex m; return m; }
inline int& fd(){ static int f=-1; return f; }
inline std::unordered_map<uintptr_t,Pending>& pend(){ static std::unordered_map<uintptr_t,Pending> m; return m; }
inline std::unordered_map<uintptr_t,Slice*>& built(){ static std::unordered_map<uintptr_t,Slice*> m; return m; }
inline std::unordered_map<uintptr_t,RemoteFull>& rfull(){ static std::unordered_map<uintptr_t,RemoteFull> m; return m; }

inline bool _sendall(int f,const void*p,size_t n){const char*c=(const char*)p;while(n){ssize_t k=send(f,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}
inline bool _recvall(int f,void*p,size_t n){char*c=(char*)p;while(n){ssize_t k=recv(f,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}

inline int state(){ // -1 uninit, 0 disabled, 1 enabled
    static int s=-1;
    if(s!=-1) return s;
    const char* e=getenv("RKNPU_TP_SHARD"); // "ip:port"
    if(!e||!*e){ s=0; return s; }
    std::string ep=e; auto c=ep.find(':'); std::string ip=ep.substr(0,c); int port=atoi(ep.substr(c+1).c_str());
    int f=socket(AF_INET,SOCK_STREAM,0); int one=1;
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,ip.c_str(),&sa.sin_addr);
    if(connect(f,(sockaddr*)&sa,sizeof(sa))<0){ fprintf(stderr,"[rktp] connect %s FAILED - TP disabled\n",e); close(f); s=0; return s; }
    setsockopt(f,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    fd()=f; s=1; fprintf(stderr,"[rktp] connected shard %s - TP ENABLED (min_n=%d locfrac=%.2f)\n",e,TP_MIN_N,loc_frac());
    return s;
}
inline bool enabled(){ return state()==1; }

inline bool row_ok(const struct ggml_tensor* t){
    if(in_build()) return false;
    if(t->ne[2]!=1||t->ne[3]!=1) return false;
    int N=(int)t->ne[1], K=(int)t->ne[0];
    if(N<TP_MIN_N) return false;
    if(N%TP_ALIGN!=0||K%32!=0) return false;
    return true;
}
// non-row-splittable matmul weights offloaded WHOLE to the shard (round-robin by name hash).
inline bool whole_ok(const struct ggml_tensor* t){
    if(in_build()) return false;
    if(offload_frac()<=0.0) return false;
    if(t->ne[2]!=1||t->ne[3]!=1) return false;
    int N=(int)t->ne[1], K=(int)t->ne[0];
    if(N<64||K%32!=0||N%16!=0) return false;   // shard requant alignment; skip tiny
    if(row_ok(t)) return false;                 // handled by row-split
    return name_bucket(t->name) < (unsigned)(offload_frac()*1000.0);
}
inline bool qualifies(const struct ggml_tensor* t){ return row_ok(t)||whole_ok(t); }

// compute Nloc/Nrem for a given N given locfrac; returns false if not splittable
inline bool split_dims(int N,int& Nloc,int& Nrem){
    Nloc=((int)(N*loc_frac())/TP_ALIGN)*TP_ALIGN;
    if(Nloc<TP_ALIGN) Nloc=TP_ALIGN;
    Nrem=N-Nloc;
    if(Nrem<16||Nrem%16!=0) return false;
    return true;
}

// called from RKNPU set_tensor for a qualifying weight (full-tensor set). ships remote slice,
// eager-builds local slice (frees raw immediately). Falls back to lazy pend if backend not captured.
inline bool on_set_tensor(const struct ggml_tensor* t,const void* data,size_t offset,size_t size){
    int K=(int)t->ne[0], N=(int)t->ne[1];
    size_t row=ggml_row_size(t->type,K);
    if(offset!=0 || size!=(size_t)N*row) return false; // only whole-tensor sets
    uintptr_t id=(uintptr_t)t->data;
    const uint8_t* d=(const uint8_t*)data;
    // WHOLE-OFFLOAD: non-row-splittable matmul -> ship ALL N rows to shard, keep nothing local.
    if(!row_ok(t)){
        std::lock_guard<std::mutex> lk(mx());
        uint8_t cmd=1; uint64_t rid=id; int32_t ty=(int)t->type,kk=K,nn=N; uint64_t nb=(uint64_t)N*row;
        if(!_sendall(fd(),&cmd,1)||!_sendall(fd(),&rid,8)||!_sendall(fd(),&ty,4)||!_sendall(fd(),&kk,4)||!_sendall(fd(),&nn,4)||!_sendall(fd(),&nb,8)||!_sendall(fd(),d,nb)){ fprintf(stderr,"[rktp] WHOLE send fail\n"); return false; }
        int32_t ok=0; _recvall(fd(),&ok,4);
        rfull()[id]=RemoteFull{K,N,(int)t->type};
        if(getenv("RKNPU_TP_DEBUG")) fprintf(stderr,"[rktp] WHOLE  name=%s K=%d N=%d MB=%.1f\n",t->name,K,N,(double)nb/1048576.0);
        drop_src(data,size);
        return true;
    }
    // ROW-SPLIT: keep local rows [0,Nloc), ship remote rows [Nloc,N).
    int Nloc=0,Nrem=0; if(!split_dims(N,Nloc,Nrem)) return false;
    std::lock_guard<std::mutex> lk(mx());
    uint8_t cmd=1; uint64_t rid=id; int32_t ty=(int)t->type,kk=K,nn=Nrem; uint64_t nb=(uint64_t)Nrem*row;
    if(!_sendall(fd(),&cmd,1)||!_sendall(fd(),&rid,8)||!_sendall(fd(),&ty,4)||!_sendall(fd(),&kk,4)||!_sendall(fd(),&nn,4)||!_sendall(fd(),&nb,8)||!_sendall(fd(),d+(size_t)Nloc*row,nb)){ fprintf(stderr,"[rktp] LOAD send fail\n"); return false; }
    int32_t ok=0; _recvall(fd(),&ok,4);
    if(g_be()){
        // EAGER: requant local half into an NPU slice now; do not retain raw bytes.
        in_build()=true; Slice* sl=new Slice(); sl->build(g_be(),(int)t->type,K,Nloc,d,(size_t)Nloc*row); in_build()=false;
        built()[id]=sl;
    } else {
        // fallback lazy path (backend not captured): keep local raw for build at first compute
        Pending p; p.type=(int)t->type; p.K=K; p.Nloc=Nloc; p.Nrem=Nrem; p.N=N;
        p.loc.assign(d, d+(size_t)Nloc*row);
        pend()[id]=std::move(p);
    }
    drop_src(data,size);   // release the full source weight from coordinator file-mmap
    return true;
}

inline bool is_split(const struct ggml_tensor* t){
    uintptr_t id=(uintptr_t)t->data; std::lock_guard<std::mutex> lk(mx());
    return pend().count(id)||built().count(id)||rfull().count(id);
}

// distributed matmul: dst[N,M] = concat(local[Nloc], remote[Nrem]) ; x is [K,M] f32
inline void compute(ggml_backend_t be,const struct ggml_tensor* src0,const float* x,int M,float* dst){
    uintptr_t id=(uintptr_t)src0->data;
    // WHOLE-OFFLOAD: entire tensor computed on shard; send x, receive full [M,N].
    { std::lock_guard<std::mutex> lk(mx());
      auto rf=rfull().find(id);
      if(rf!=rfull().end()){ int K=rf->second.K;
          uint8_t cmd=2; uint64_t rid=id; int32_t mm=M,kk=K; uint64_t nb=(uint64_t)M*K*4;
          _sendall(fd(),&cmd,1);_sendall(fd(),&rid,8);_sendall(fd(),&mm,4);_sendall(fd(),&kk,4);_sendall(fd(),&nb,8);_sendall(fd(),x,nb);
          uint64_t ob=0; _recvall(fd(),&ob,8); if(ob) _recvall(fd(),dst,ob);
          return;
      }
    }
    Slice* sl=nullptr; int K=0;
    { std::lock_guard<std::mutex> lk(mx());
      auto b=built().find(id);
      if(b!=built().end()){ sl=b->second; }
      else{ auto p=pend().find(id); Pending pp=p->second; pend().erase(p);
            in_build()=true; sl=new Slice(); sl->build(be,pp.type,pp.K,pp.Nloc,pp.loc.data(),pp.loc.size()); in_build()=false;
            built()[id]=sl; }
      K=sl->K; }
    int Nloc=sl->N; int N=(int)src0->ne[1]; int Nrem=N-Nloc;
    std::vector<float> orem((size_t)M*Nrem), oloc((size_t)M*Nloc);
    // NO thread-per-matmul: send x (TCP-buffered -> returns fast, shard computes remote
    // concurrently), run local slice on THIS NPU meanwhile, then recv the remote result.
    { std::lock_guard<std::mutex> lk(mx());
        uint8_t cmd=2; uint64_t rid=id; int32_t mm=M,kk=K; uint64_t nb=(uint64_t)M*K*4;
        _sendall(fd(),&cmd,1);_sendall(fd(),&rid,8);_sendall(fd(),&mm,4);_sendall(fd(),&kk,4);_sendall(fd(),&nb,8);_sendall(fd(),x,nb);
    }
    sl->run(M,x,oloc.data());     // local slice on THIS NPU  ||  remote computing on shard
    { std::lock_guard<std::mutex> lk(mx());
        uint64_t ob=0; _recvall(fd(),&ob,8); if(ob) _recvall(fd(),orem.data(),ob);
    }
    for(int m=0;m<M;m++){ float* dr=dst+(size_t)m*N; const float* lr=oloc.data()+(size_t)m*Nloc; const float* rr=orem.data()+(size_t)m*Nrem;
        for(int n=0;n<Nloc;n++) dr[n]=lr[n];
        for(int n=0;n<Nrem;n++) dr[Nloc+n]=rr[n]; }
}

inline void Slice::build(ggml_backend_t b,int t,int K_,int N_,const void* bytes,size_t nb){
    be=b; type=t; K=K_; N=N_;
    ggml_init_params p0{ ggml_tensor_overhead()*4, nullptr, true }; cw=ggml_init(p0);
    W=ggml_new_tensor_2d(cw,(ggml_type)t,K,N); bw=ggml_backend_alloc_ctx_tensors(cw,be);
    ggml_backend_tensor_set(W,bytes,0,nb);
}
inline void Slice::run(int M,const float* xd,float* od){
    if(M!=curM){ if(al){ggml_gallocr_free(al);al=nullptr;} if(cc){ggml_free(cc);cc=nullptr;}
        ggml_init_params pc{ ggml_tensor_overhead()*8+ggml_graph_overhead(), nullptr, true }; cc=ggml_init(pc);
        gf=ggml_new_graph(cc); x=ggml_new_tensor_2d(cc,GGML_TYPE_F32,K,M); o=ggml_mul_mat(cc,W,x);
        ggml_build_forward_expand(gf,o); al=ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(al,gf); curM=M; }
    ggml_backend_tensor_set(x,xd,0,(size_t)K*M*sizeof(float));
    ggml_backend_graph_compute(be,gf);
    ggml_backend_tensor_get(o,od,0,(size_t)N*M*sizeof(float));
}

} // namespace rktp
