// rktp.h - RKNPU tensor-parallel (row-split) client + local-slice engine.
// Split a big MUL_MAT weight by output-rows across the LOCAL NPU + N shard NPUs
// (tp_shard daemons on other boards); all compute concurrently, then concat.
// v3: N-shard (RKNPU_TP_SHARD = comma list "ip1:port1,ip2:port2"); reduces to the
//     proven 2-board path when 1 shard. eager local-slice build; RKNPU_TP_LOCFRAC
//     local fraction; RKNPU_TP_OFFLOAD whole-tensor offload of non-row-splittable mms.
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
#include <string>
#include <sstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/mman.h>

namespace rktp {

// Release source (gguf-mmap) weight pages after requant/ship (env RKNPU_TP_DROPSRC).
inline void drop_src(const void* data, size_t size){
    static int on = -1; if(on==-1) on = getenv("RKNPU_TP_DROPSRC")?1:0;
    if(!on || !data || !size) return;
    long ps = sysconf(_SC_PAGESIZE); if(ps<=0) return;
    uintptr_t start=(uintptr_t)data, end=start+size;
    uintptr_t a=(start+ps-1)&~((uintptr_t)ps-1);
    uintptr_t b=end&~((uintptr_t)ps-1);
    if(b>a) madvise((void*)a,(size_t)(b-a),MADV_DONTNEED);
}

static const int   TP_ALIGN = 64;
inline int tp_min_n(){ static int v = (getenv("RKNPU_TP_MINN")? atoi(getenv("RKNPU_TP_MINN")) : 8192); return v; }
#define TP_MIN_N tp_min_n()
inline double loc_frac(){ static double v = (getenv("RKNPU_TP_LOCFRAC")? atof(getenv("RKNPU_TP_LOCFRAC")) : 0.5); if(v<=0.02||v>=0.95) v=0.5; return v; }
inline double offload_frac(){ static double v = (getenv("RKNPU_TP_OFFLOAD")? atof(getenv("RKNPU_TP_OFFLOAD")) : 0.0); if(v<0) v=0; if(v>1) v=1; return v; }
inline unsigned name_bucket(const char* n){ unsigned h=2166136261u; if(n) for(const char*p=n;*p;++p){h^=(unsigned char)*p;h*=16777619u;} return h%1000; }

inline ggml_backend_t& g_be(){ static ggml_backend_t b=nullptr; return b; }

struct Slice {
    ggml_backend_t be=nullptr; ggml_context* cw=nullptr; ggml_backend_buffer_t bw=nullptr; ggml_tensor* W=nullptr;
    int type=0,K=0,N=0,curM=-1; ggml_context* cc=nullptr; ggml_gallocr_t al=nullptr; ggml_cgraph* gf=nullptr; ggml_tensor *x=nullptr,*o=nullptr;
    void build(ggml_backend_t b,int t,int K_,int N_,const void* bytes,size_t nb);
    void run(int M,const float* xd,float* od);
};

struct Pending { int type,K,Nloc,Nrem,N; std::vector<uint8_t> loc; };  // fallback lazy path
struct RemoteFull { int K,N,type,shard; };  // whole tensor offloaded to shard[shard]

inline bool& in_build(){ static thread_local bool v=false; return v; }
inline std::mutex& mx(){ static std::mutex m; return m; }
inline std::vector<int>& fds(){ static std::vector<int> v; return v; }   // shard sockets
inline int nsh(){ return (int)fds().size(); }
inline std::unordered_map<uintptr_t,Pending>& pend(){ static std::unordered_map<uintptr_t,Pending> m; return m; }
inline std::unordered_map<uintptr_t,Slice*>& built(){ static std::unordered_map<uintptr_t,Slice*> m; return m; }
inline std::unordered_map<uintptr_t,RemoteFull>& rfull(){ static std::unordered_map<uintptr_t,RemoteFull> m; return m; }

inline bool _sendall(int f,const void*p,size_t n){const char*c=(const char*)p;while(n){ssize_t k=send(f,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}
inline bool _recvall(int f,void*p,size_t n){char*c=(char*)p;while(n){ssize_t k=recv(f,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}

inline int state(){ // -1 uninit, 0 disabled, 1 enabled
    static int s=-1;
    if(s!=-1) return s;
    const char* e=getenv("RKNPU_TP_SHARD"); // "ip:port[,ip:port...]"
    if(!e||!*e){ s=0; return s; }
    std::stringstream ss(e); std::string ep;
    while(std::getline(ss,ep,',')){
        if(ep.empty()) continue;
        auto c=ep.find(':'); std::string ip=ep.substr(0,c); int port=atoi(ep.substr(c+1).c_str());
        int f=socket(AF_INET,SOCK_STREAM,0); int one=1;
        sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,ip.c_str(),&sa.sin_addr);
        if(connect(f,(sockaddr*)&sa,sizeof(sa))<0){ fprintf(stderr,"[rktp] connect %s FAILED - TP disabled\n",ep.c_str()); close(f); fds().clear(); s=0; return s; }
        setsockopt(f,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
        fds().push_back(f);
    }
    if(fds().empty()){ s=0; return s; }
    s=1; fprintf(stderr,"[rktp] connected %d shard(s) - TP ENABLED (min_n=%d locfrac=%.2f offload=%.2f)\n",nsh(),TP_MIN_N,loc_frac(),offload_frac());
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
inline bool whole_ok(const struct ggml_tensor* t){
    if(in_build()) return false;
    if(offload_frac()<=0.0) return false;
    if(t->ne[2]!=1||t->ne[3]!=1) return false;
    int N=(int)t->ne[1], K=(int)t->ne[0];
    if(N<64||K%32!=0||N%16!=0) return false;
    if(row_ok(t)) return false;
    return name_bucket(t->name) < (unsigned)(offload_frac()*1000.0);
}
inline bool qualifies(const struct ggml_tensor* t){ return row_ok(t)||whole_ok(t); }

// Split N output-rows into (nsh+1) TP_ALIGN-multiple parts: [local, shard0, shard1, ...].
// local gets loc_frac; the rest split ~equally among shards (last shard takes remainder).
inline std::vector<int> compute_parts(int N){
    int P = nsh()+1; std::vector<int> parts(P,0);
    int Nloc=((int)(N*loc_frac())/TP_ALIGN)*TP_ALIGN; if(Nloc<TP_ALIGN) Nloc=TP_ALIGN;
    if(Nloc>=N){ parts.clear(); return parts; }
    parts[0]=Nloc; int rem=N-Nloc; int ns=P-1;
    int per=((rem/ns)/TP_ALIGN)*TP_ALIGN; if(per<TP_ALIGN) per=TP_ALIGN;
    int used=0; for(int i=1;i<P-1;i++){ parts[i]=per; used+=per; }
    parts[P-1]=rem-used;
    for(int p:parts) if(p<TP_ALIGN){ parts.clear(); return parts; }  // invalid split
    return parts;
}

// keep split_dims for get_alloc_size: local slice size vs total remote
inline bool split_dims(int N,int& Nloc,int& Nrem){
    auto parts=compute_parts(N); if(parts.empty()) return false;
    Nloc=parts[0]; Nrem=N-Nloc; return true;
}

// ship rows [r0,r0+len) of tensor id to shard sfd (LOAD cmd=1)
inline bool _ship(int sfd,uint64_t rid,int type,int K,int rows,size_t row,const uint8_t* base,int r0){
    uint8_t cmd=1; int32_t ty=type,kk=K,nn=rows; uint64_t nb=(uint64_t)rows*row;
    if(!_sendall(sfd,&cmd,1)||!_sendall(sfd,&rid,8)||!_sendall(sfd,&ty,4)||!_sendall(sfd,&kk,4)||!_sendall(sfd,&nn,4)||!_sendall(sfd,&nb,8)||!_sendall(sfd,base+(size_t)r0*row,nb)){ fprintf(stderr,"[rktp] ship fail\n"); return false; }
    int32_t ok=0; _recvall(sfd,&ok,4); return true;
}

inline bool on_set_tensor(const struct ggml_tensor* t,const void* data,size_t offset,size_t size){
    int K=(int)t->ne[0], N=(int)t->ne[1];
    size_t row=ggml_row_size(t->type,K);
    if(offset!=0 || size!=(size_t)N*row) return false;
    uintptr_t id=(uintptr_t)t->data;
    const uint8_t* d=(const uint8_t*)data;
    // WHOLE-OFFLOAD -> one shard (round-robin by name)
    if(!row_ok(t)){
        int sh=(int)(name_bucket(t->name)%(unsigned)nsh());
        std::lock_guard<std::mutex> lk(mx());
        if(!_ship(fds()[sh],id,(int)t->type,K,N,row,d,0)) return false;
        rfull()[id]=RemoteFull{K,N,(int)t->type,sh};
        if(getenv("RKNPU_TP_DEBUG")) fprintf(stderr,"[rktp] WHOLE name=%s K=%d N=%d ->shard%d MB=%.1f\n",t->name,K,N,sh,(double)N*row/1048576.0);
        drop_src(data,size); return true;
    }
    // ROW-SPLIT across local + shards
    auto parts=compute_parts(N); if(parts.empty()) return false;   // not splittable -> normal requant on coord
    int Nloc=parts[0];
    std::lock_guard<std::mutex> lk(mx());
    int off=Nloc;
    for(int i=0;i<nsh();i++){ int len=parts[i+1]; if(!_ship(fds()[i],id,(int)t->type,K,len,row,d,off)) return false; off+=len; }
    if(g_be()){
        in_build()=true; Slice* sl=new Slice(); sl->build(g_be(),(int)t->type,K,Nloc,d,(size_t)Nloc*row); in_build()=false;
        built()[id]=sl;
    } else {
        Pending p; p.type=(int)t->type; p.K=K; p.Nloc=Nloc; p.Nrem=N-Nloc; p.N=N;
        p.loc.assign(d, d+(size_t)Nloc*row); pend()[id]=std::move(p);
    }
    drop_src(data,size); return true;
}

inline bool is_split(const struct ggml_tensor* t){
    uintptr_t id=(uintptr_t)t->data; std::lock_guard<std::mutex> lk(mx());
    return pend().count(id)||built().count(id)||rfull().count(id);
}

inline void compute(ggml_backend_t be,const struct ggml_tensor* src0,const float* x,int M,float* dst){
    uintptr_t id=(uintptr_t)src0->data;
    // WHOLE-OFFLOAD: full tensor on one shard
    { std::lock_guard<std::mutex> lk(mx());
      auto rf=rfull().find(id);
      if(rf!=rfull().end()){ int K=rf->second.K; int f=fds()[rf->second.shard];
          uint8_t cmd=2; uint64_t rid=id; int32_t mm=M,kk=K; uint64_t nb=(uint64_t)M*K*4;
          _sendall(f,&cmd,1);_sendall(f,&rid,8);_sendall(f,&mm,4);_sendall(f,&kk,4);_sendall(f,&nb,8);_sendall(f,x,nb);
          uint64_t ob=0; _recvall(f,&ob,8); if(ob) _recvall(f,dst,ob);
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
    int Nloc=sl->N; int N=(int)src0->ne[1];
    auto parts=compute_parts(N);   // deterministic; parts[0]==Nloc
    int ns=nsh();
    std::vector<float> oloc((size_t)M*Nloc);
    std::vector<std::vector<float>> orem(ns);
    // send x to ALL shards (TCP-buffered -> they compute concurrently)
    { std::lock_guard<std::mutex> lk(mx());
      for(int i=0;i<ns;i++){ int f=fds()[i]; uint8_t cmd=2; uint64_t rid=id; int32_t mm=M,kk=K; uint64_t nb=(uint64_t)M*K*4;
          _sendall(f,&cmd,1);_sendall(f,&rid,8);_sendall(f,&mm,4);_sendall(f,&kk,4);_sendall(f,&nb,8);_sendall(f,x,nb); }
    }
    sl->run(M,x,oloc.data());     // local slice while shards compute
    { std::lock_guard<std::mutex> lk(mx());
      for(int i=0;i<ns;i++){ int f=fds()[i]; uint64_t ob=0; _recvall(f,&ob,8); orem[i].resize(ob/4); if(ob) _recvall(f,orem[i].data(),ob); }
    }
    // concat: [local | shard0 | shard1 | ...] per token
    for(int m=0;m<M;m++){ float* dr=dst+(size_t)m*N;
        const float* lr=oloc.data()+(size_t)m*Nloc;
        for(int n=0;n<Nloc;n++) dr[n]=lr[n];
        int off=Nloc;
        for(int i=0;i<ns;i++){ int len=parts[i+1]; const float* rr=orem[i].data()+(size_t)m*len;
            for(int n=0;n<len;n++) dr[off+n]=rr[n]; off+=len; }
    }
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
