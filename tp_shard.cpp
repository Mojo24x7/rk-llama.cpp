// tp_shard: distributed NPU matmul shard + protocol.
//  roles:
//   shard  -> daemon: LOAD(id,type,K,N,bytes) build weight-slice matmul on NPU; MM(id,M,x)->partial
//   coord  -> self-test: synth full W[K,N] F16; local half via NpuMM; ship upper half to shard; MM; compare concat vs full.
// Protocol (little-endian, TCP, NODELAY):
//   LOAD: u8=1, u64 id, i32 type, i32 K, i32 N, u64 nbytes, bytes[nbytes]        ; reply i32 ok
//   MM:   u8=2, u64 id, i32 M, i32 K, u64 nbytes=M*K*4, x f32[M*K]               ; reply u64 nbytes=M*N*4, C f32[M*N]
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration<double,std::milli>(clk::now()-t).count(); }
static bool sendall(int fd,const void*p,size_t n){const char*c=(const char*)p;while(n){ssize_t k=send(fd,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}
static bool recvall(int fd,void*p,size_t n){char*c=(char*)p;while(n){ssize_t k=recv(fd,c,n,0);if(k<=0)return false;c+=k;n-=(size_t)k;}return true;}

static ggml_backend_t init_npu(){
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RKNPU");
    if(!reg){ fprintf(stderr,"no RKNPU reg\n"); exit(1);}
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg,0);
    ggml_backend_t b = ggml_backend_dev_init(dev,nullptr);
    if(!b){ fprintf(stderr,"RKNPU init failed\n"); exit(1);}
    fprintf(stderr,"[npu] backend=%s\n", ggml_backend_name(b));
    return b;
}

// a weight-slice matmul living on the NPU (real ggml type via RKNPU set_tensor requant)
struct NpuMM {
    ggml_backend_t backend=nullptr;
    ggml_context* ctx_w=nullptr; ggml_backend_buffer_t buf_w=nullptr; ggml_tensor* W=nullptr;
    int type=0,K=0,N=0;
    int curM=-1; ggml_context* ctx_c=nullptr; ggml_gallocr_t alloc=nullptr; ggml_cgraph* gf=nullptr; ggml_tensor *x=nullptr,*out=nullptr;
    void load(ggml_backend_t b,int type_,int K_,int N_,const void* bytes,size_t nbytes){
        backend=b; type=type_; K=K_; N=N_;
        ggml_init_params p0{ ggml_tensor_overhead()*4, nullptr, true };
        ctx_w=ggml_init(p0);
        W=ggml_new_tensor_2d(ctx_w,(ggml_type)type,K,N);
        buf_w=ggml_backend_alloc_ctx_tensors(ctx_w,backend);
        ggml_backend_tensor_set(W,bytes,0,nbytes);
    }
    void ensureM(int M){
        if(M==curM) return;
        if(alloc){ ggml_gallocr_free(alloc); alloc=nullptr; }
        if(ctx_c){ ggml_free(ctx_c); ctx_c=nullptr; }
        ggml_init_params pc{ ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ctx_c=ggml_init(pc);
        gf=ggml_new_graph(ctx_c);
        x=ggml_new_tensor_2d(ctx_c,GGML_TYPE_F32,K,M);
        out=ggml_mul_mat(ctx_c,W,x);
        ggml_build_forward_expand(gf,out);
        alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc,gf);
        curM=M;
    }
    void run(int M,const float* xd,float* od){
        ensureM(M);
        ggml_backend_tensor_set(x,xd,0,(size_t)K*M*sizeof(float));
        ggml_backend_graph_compute(backend,gf);
        ggml_backend_tensor_get(out,od,0,(size_t)N*M*sizeof(float));
    }
};

static int run_shard(int port){
    ggml_backend_t b = init_npu();
    int ls=socket(AF_INET,SOCK_STREAM,0); int one=1;
    setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_addr.s_addr=INADDR_ANY; sa.sin_port=htons(port);
    if(bind(ls,(sockaddr*)&sa,sizeof(sa))<0){perror("bind");return 1;}
    listen(ls,1);
    fprintf(stderr,"[shard] listening :%d\n",port);
    int fd=accept(ls,nullptr,nullptr); setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    fprintf(stderr,"[shard] connected\n");
    std::unordered_map<uint64_t,NpuMM*> reg;
    std::vector<uint8_t> buf; std::vector<float> xin, cout;
    while(true){
        uint8_t cmd;
        if(!recvall(fd,&cmd,1)) break;
        if(cmd==1){ // LOAD
            uint64_t id; int32_t type,K,N; uint64_t nb;
            recvall(fd,&id,8); recvall(fd,&type,4); recvall(fd,&K,4); recvall(fd,&N,4); recvall(fd,&nb,8);
            buf.resize(nb); recvall(fd,buf.data(),nb);
            NpuMM* mm=new NpuMM(); mm->load(b,type,K,N,buf.data(),nb);
            reg[id]=mm;
            int32_t ok=1; sendall(fd,&ok,4);
            fprintf(stderr,"[shard] LOAD id=%llu type=%d K=%d N=%d nb=%llu\n",(unsigned long long)id,type,K,N,(unsigned long long)nb);
        } else if(cmd==2){ // MM
            uint64_t id; int32_t M,K; uint64_t nb;
            recvall(fd,&id,8); recvall(fd,&M,4); recvall(fd,&K,4); recvall(fd,&nb,8);
            xin.resize((size_t)M*K); recvall(fd,xin.data(),nb);
            auto it=reg.find(id);
            if(it==reg.end()){ fprintf(stderr,"[shard] MM unknown id %llu\n",(unsigned long long)id); uint64_t z=0; sendall(fd,&z,8); continue; }
            NpuMM* mm=it->second; cout.resize((size_t)M*mm->N);
            mm->run(M,xin.data(),cout.data());
            uint64_t ob=(uint64_t)M*mm->N*sizeof(float);
            sendall(fd,&ob,8); sendall(fd,cout.data(),ob);
        } else break;
    }
    fprintf(stderr,"[shard] done\n"); return 0;
}

// synthetic verifier: full F16 W[K,N]; local half rows[0,H); shard rows[H,N); compare
static float Wf(int k,int n){ return 0.05f*sinf(0.001f*k + 0.002f*n + 0.7f); }
static float xf(int k){ return sinf(0.01f*k + 0.3f); }
static int run_coord(const std::string& peer,int port,int K,int N){
    int H=N/2, R=N-H;
    ggml_backend_t b=init_npu();
    // full ref on local NPU
    NpuMM full; { std::vector<ggml_fp16_t> wd((size_t)K*N); for(int j=0;j<N;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,j)); full.load(b,GGML_TYPE_F16,K,N,wd.data(),wd.size()*2);}
    std::vector<float> x(K); for(int i=0;i<K;i++)x[i]=xf(i);
    std::vector<float> outF(N); full.run(1,x.data(),outF.data());
    // local half rows[0,H)
    NpuMM loc; { std::vector<ggml_fp16_t> wd((size_t)K*H); for(int j=0;j<H;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,j)); loc.load(b,GGML_TYPE_F16,K,H,wd.data(),wd.size()*2);}
    // connect shard, LOAD upper half rows[H,N)
    int fd=socket(AF_INET,SOCK_STREAM,0); int one=1;
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,peer.c_str(),&sa.sin_addr);
    if(connect(fd,(sockaddr*)&sa,sizeof(sa))<0){perror("connect");return 1;}
    setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
    { std::vector<ggml_fp16_t> wd((size_t)K*R); for(int j=0;j<R;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,H+j));
      uint8_t cmd=1; uint64_t id=42; int32_t t=GGML_TYPE_F16,kk=K,nn=R; uint64_t nb=wd.size()*2;
      sendall(fd,&cmd,1);sendall(fd,&id,8);sendall(fd,&t,4);sendall(fd,&kk,4);sendall(fd,&nn,4);sendall(fd,&nb,8);sendall(fd,wd.data(),nb);
      int32_t ok=0; recvall(fd,&ok,4); fprintf(stderr,"[coord] shard LOAD ok=%d\n",ok);}
    std::vector<float> out0(H),out1(R),outS(N);
    // warmup + timed split (async overlap)
    auto do_split=[&](std::vector<double>&ts,int runs){
        for(int r=0;r<runs;r++){ auto t=clk::now();
            std::thread th([&]{ uint8_t cmd=2; uint64_t id=42; int32_t M=1,kk=K; uint64_t nb=(uint64_t)K*4;
                sendall(fd,&cmd,1);sendall(fd,&id,8);sendall(fd,&M,4);sendall(fd,&kk,4);sendall(fd,&nb,8);sendall(fd,x.data(),nb);
                uint64_t ob=0; recvall(fd,&ob,8); recvall(fd,out1.data(),ob); });
            loc.run(1,x.data(),out0.data()); th.join();
            if(ts.size()<(size_t)runs) ts.push_back(ms_since(t)); }
    };
    std::vector<double> warm; do_split(warm,5); warm.clear();
    std::vector<double> ts; do_split(ts,50);
    for(int i=0;i<H;i++)outS[i]=out0[i]; for(int i=0;i<R;i++)outS[H+i]=out1[i];
    std::vector<double> tf; for(int r=0;r<50;r++){auto t=clk::now(); full.run(1,x.data(),outF.data()); tf.push_back(ms_since(t));}
    std::sort(tf.begin(),tf.end()); std::sort(ts.begin(),ts.end());
    double maxerr=0,ref=0; for(int i=0;i<N;i++){maxerr=std::max(maxerr,(double)fabs(outS[i]-outF[i]));ref=std::max(ref,(double)fabs(outF[i]));}
    printf("PROTOCOL TEST K=%d N=%d  FULL=%.3fms SPLIT=%.3fms speedup=%.2fx  max|err|=%.4g (ref=%.4g)\n",
           K,N,tf[25],ts[25],tf[25]/ts[25],maxerr,ref);
    uint8_t stop=0; sendall(fd,&stop,1); close(fd);
    return 0;
}


// 2-shard synthetic verifier: local[0,H0) + shardA[H0,H0+R1) + shardB[H0+R1,N); compare concat vs full.
static int run_coord2(const std::string& pA,const std::string& pB,int port,int K,int N){
    int H0=(N/3/64)*64; int R1=(N/3/64)*64; int R2=N-H0-R1;
    ggml_backend_t b=init_npu();
    NpuMM full; { std::vector<ggml_fp16_t> wd((size_t)K*N); for(int j=0;j<N;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,j)); full.load(b,GGML_TYPE_F16,K,N,wd.data(),wd.size()*2);}
    std::vector<float> x(K); for(int i=0;i<K;i++)x[i]=xf(i);
    std::vector<float> outF(N); full.run(1,x.data(),outF.data());
    NpuMM loc; { std::vector<ggml_fp16_t> wd((size_t)K*H0); for(int j=0;j<H0;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,j)); loc.load(b,GGML_TYPE_F16,K,H0,wd.data(),wd.size()*2);}
    auto conn=[&](const std::string& ip)->int{ int fd=socket(AF_INET,SOCK_STREAM,0); int one=1; sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,ip.c_str(),&sa.sin_addr); if(connect(fd,(sockaddr*)&sa,sizeof(sa))<0){perror("connect");exit(1);} setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one)); return fd; };
    int fA=conn(pA), fB=conn(pB);
    auto ship=[&](int fd,int r0,int R){ std::vector<ggml_fp16_t> wd((size_t)K*R); for(int j=0;j<R;j++)for(int i=0;i<K;i++)wd[(size_t)j*K+i]=ggml_fp32_to_fp16(Wf(i,r0+j)); uint8_t cmd=1; uint64_t id=42; int32_t t=GGML_TYPE_F16,kk=K,nn=R; uint64_t nb=wd.size()*2; sendall(fd,&cmd,1);sendall(fd,&id,8);sendall(fd,&t,4);sendall(fd,&kk,4);sendall(fd,&nn,4);sendall(fd,&nb,8);sendall(fd,wd.data(),nb); int32_t ok=0; recvall(fd,&ok,4); };
    ship(fA,H0,R1); ship(fB,H0+R1,R2);
    std::vector<float> o0(H0),oA(R1),oB(R2),outS(N);
    { uint8_t cmd=2; uint64_t id=42; int32_t M=1,kk=K; uint64_t nb=(uint64_t)K*4;
      sendall(fA,&cmd,1);sendall(fA,&id,8);sendall(fA,&M,4);sendall(fA,&kk,4);sendall(fA,&nb,8);sendall(fA,x.data(),nb);
      sendall(fB,&cmd,1);sendall(fB,&id,8);sendall(fB,&M,4);sendall(fB,&kk,4);sendall(fB,&nb,8);sendall(fB,x.data(),nb); }
    loc.run(1,x.data(),o0.data());
    { uint64_t ob=0; recvall(fA,&ob,8); recvall(fA,oA.data(),ob); ob=0; recvall(fB,&ob,8); recvall(fB,oB.data(),ob); }
    for(int i=0;i<H0;i++)outS[i]=o0[i]; for(int i=0;i<R1;i++)outS[H0+i]=oA[i]; for(int i=0;i<R2;i++)outS[H0+R1+i]=oB[i];
    double maxerr=0,ref=0; int bad=-1; for(int i=0;i<N;i++){double e=fabs(outS[i]-outF[i]); if(e>maxerr){maxerr=e;bad=i;} ref=std::max(ref,(double)fabs(outF[i]));}
    const char* rg = bad<H0?"LOCAL":(bad<H0+R1?"shardA":"shardB");
    printf("2SHARD TEST K=%d N=%d H0=%d R1=%d R2=%d  max|err|=%.4g (ref=%.4g) worst_i=%d region=%s\n",K,N,H0,R1,R2,maxerr,ref,bad,rg);
    uint8_t stop=0; sendall(fA,&stop,1); sendall(fB,&stop,1); close(fA); close(fB);
    return 0;
}

int main(int argc,char**argv){
    std::string role,peer="192.168.1.10",peerB="192.168.1.11"; int port=48200,K=5120,N=13824;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--role"&&i+1<argc)role=argv[++i];
        else if(a=="--peer"&&i+1<argc)peer=argv[++i];
        else if(a=="--peerB"&&i+1<argc)peerB=argv[++i];
        else if(a=="--port"&&i+1<argc)port=atoi(argv[++i]);
        else if(a=="--K"&&i+1<argc)K=atoi(argv[++i]);
        else if(a=="--N"&&i+1<argc)N=atoi(argv[++i]);}
    if(role=="shard") return run_shard(port);
    if(role=="coord2") return run_coord2(peer,peerB,port,K,N);
    return run_coord(peer,port,K,N);
}
