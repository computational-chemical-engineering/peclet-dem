#include <mpi.h>
#include <cstdio>
#include <cstdlib>
static double T[64]; static long N[64];
static const char* NM[]={"Allreduce","Waitall","Waitany","Wait","Isend","Issend","Irecv","Testall","Test","Sendrecv","Recv","Send","Neighbor_alltoallv","Iprobe","Ibarrier","Barrier","Exscan","Allgather","Allgatherv","Alltoall","Alltoallv","Bcast"};
extern "C" int MPI_Allreduce(const void* a, void* b, int c, MPI_Datatype d, MPI_Op e, MPI_Comm f){double t=PMPI_Wtime();int r=PMPI_Allreduce(a,b,c,d,e,f);T[0]+=PMPI_Wtime()-t;N[0]++;return r;}
extern "C" int MPI_Waitall(int a, MPI_Request* b, MPI_Status* c){double t=PMPI_Wtime();int r=PMPI_Waitall(a,b,c);T[1]+=PMPI_Wtime()-t;N[1]++;return r;}
extern "C" int MPI_Waitany(int a, MPI_Request* b, int* c, MPI_Status* d){double t=PMPI_Wtime();int r=PMPI_Waitany(a,b,c,d);T[2]+=PMPI_Wtime()-t;N[2]++;return r;}
extern "C" int MPI_Wait(MPI_Request* a, MPI_Status* b){double t=PMPI_Wtime();int r=PMPI_Wait(a,b);T[3]+=PMPI_Wtime()-t;N[3]++;return r;}
extern "C" int MPI_Isend(const void* a, int b, MPI_Datatype c, int d, int e, MPI_Comm f, MPI_Request* g){double t=PMPI_Wtime();int r=PMPI_Isend(a,b,c,d,e,f,g);T[4]+=PMPI_Wtime()-t;N[4]++;return r;}
extern "C" int MPI_Issend(const void* a, int b, MPI_Datatype c, int d, int e, MPI_Comm f, MPI_Request* g){double t=PMPI_Wtime();int r=PMPI_Issend(a,b,c,d,e,f,g);T[5]+=PMPI_Wtime()-t;N[5]++;return r;}
extern "C" int MPI_Irecv(void* a, int b, MPI_Datatype c, int d, int e, MPI_Comm f, MPI_Request* g){double t=PMPI_Wtime();int r=PMPI_Irecv(a,b,c,d,e,f,g);T[6]+=PMPI_Wtime()-t;N[6]++;return r;}
extern "C" int MPI_Testall(int a, MPI_Request* b, int* c, MPI_Status* d){double t=PMPI_Wtime();int r=PMPI_Testall(a,b,c,d);T[7]+=PMPI_Wtime()-t;N[7]++;return r;}
extern "C" int MPI_Test(MPI_Request* a, int* b, MPI_Status* c){double t=PMPI_Wtime();int r=PMPI_Test(a,b,c);T[8]+=PMPI_Wtime()-t;N[8]++;return r;}
extern "C" int MPI_Sendrecv(const void* a, int b, MPI_Datatype c, int d, int e, void* f, int g, MPI_Datatype h, int i, int j, MPI_Comm k, MPI_Status* l){double t=PMPI_Wtime();int r=PMPI_Sendrecv(a,b,c,d,e,f,g,h,i,j,k,l);T[9]+=PMPI_Wtime()-t;N[9]++;return r;}
extern "C" int MPI_Recv(void* a, int b, MPI_Datatype c, int d, int e, MPI_Comm f, MPI_Status* g){double t=PMPI_Wtime();int r=PMPI_Recv(a,b,c,d,e,f,g);T[10]+=PMPI_Wtime()-t;N[10]++;return r;}
extern "C" int MPI_Send(const void* a, int b, MPI_Datatype c, int d, int e, MPI_Comm f){double t=PMPI_Wtime();int r=PMPI_Send(a,b,c,d,e,f);T[11]+=PMPI_Wtime()-t;N[11]++;return r;}
extern "C" int MPI_Neighbor_alltoallv(const void* a, const int* b, const int* c, MPI_Datatype d, void* e, const int* f, const int* g, MPI_Datatype h, MPI_Comm i){double t=PMPI_Wtime();int r=PMPI_Neighbor_alltoallv(a,b,c,d,e,f,g,h,i);T[12]+=PMPI_Wtime()-t;N[12]++;return r;}
extern "C" int MPI_Iprobe(int a, int b, MPI_Comm c, int* d, MPI_Status* e){double t=PMPI_Wtime();int r=PMPI_Iprobe(a,b,c,d,e);T[13]+=PMPI_Wtime()-t;N[13]++;return r;}
extern "C" int MPI_Ibarrier(MPI_Comm a, MPI_Request* b){double t=PMPI_Wtime();int r=PMPI_Ibarrier(a,b);T[14]+=PMPI_Wtime()-t;N[14]++;return r;}
extern "C" int MPI_Barrier(MPI_Comm a){double t=PMPI_Wtime();int r=PMPI_Barrier(a);T[15]+=PMPI_Wtime()-t;N[15]++;return r;}
extern "C" int MPI_Exscan(const void* a, void* b, int c, MPI_Datatype d, MPI_Op e, MPI_Comm f){double t=PMPI_Wtime();int r=PMPI_Exscan(a,b,c,d,e,f);T[16]+=PMPI_Wtime()-t;N[16]++;return r;}
extern "C" int MPI_Allgather(const void* a, int b, MPI_Datatype c, void* d, int e, MPI_Datatype f, MPI_Comm g){double t=PMPI_Wtime();int r=PMPI_Allgather(a,b,c,d,e,f,g);T[17]+=PMPI_Wtime()-t;N[17]++;return r;}
extern "C" int MPI_Allgatherv(const void* a, int b, MPI_Datatype c, void* d, const int* e, const int* f, MPI_Datatype g, MPI_Comm h){double t=PMPI_Wtime();int r=PMPI_Allgatherv(a,b,c,d,e,f,g,h);T[18]+=PMPI_Wtime()-t;N[18]++;return r;}
extern "C" int MPI_Alltoall(const void* a, int b, MPI_Datatype c, void* d, int e, MPI_Datatype f, MPI_Comm g){double t=PMPI_Wtime();int r=PMPI_Alltoall(a,b,c,d,e,f,g);T[19]+=PMPI_Wtime()-t;N[19]++;return r;}
extern "C" int MPI_Alltoallv(const void* a, const int* b, const int* c, MPI_Datatype d, void* e, const int* f, const int* g, MPI_Datatype h, MPI_Comm i){double t=PMPI_Wtime();int r=PMPI_Alltoallv(a,b,c,d,e,f,g,h,i);T[20]+=PMPI_Wtime()-t;N[20]++;return r;}
extern "C" int MPI_Bcast(void* a, int b, MPI_Datatype c, int d, MPI_Comm e){double t=PMPI_Wtime();int r=PMPI_Bcast(a,b,c,d,e);T[21]+=PMPI_Wtime()-t;N[21]++;return r;}
static double t0; extern "C" int MPI_Init(int* a,char*** b){int r=PMPI_Init(a,b);t0=PMPI_Wtime();return r;}
extern "C" int MPI_Init_thread(int* a,char*** b,int c,int* d){int r=PMPI_Init_thread(a,b,c,d);t0=PMPI_Wtime();return r;}
extern "C" int MPI_Finalize(){int rk,sz;PMPI_Comm_rank(MPI_COMM_WORLD,&rk);PMPI_Comm_size(MPI_COMM_WORLD,&sz);double w=PMPI_Wtime()-t0;
 double TT[64],TM[64];PMPI_Reduce(T,TT,22,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);PMPI_Reduce(T,TM,22,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
 if(rk==0){double s=0;for(int i=0;i<22;i++){if(N[i])std::fprintf(stderr,"PMPI %-20s calls/rank0 %8ld  mean %8.3f s  max %8.3f s\n",NM[i],N[i],TT[i]/sz,TM[i]);s+=TT[i]/sz;}std::fprintf(stderr,"PMPI total_mean %.3f s wall %.3f s\n",s,w);}
 return PMPI_Finalize();}
