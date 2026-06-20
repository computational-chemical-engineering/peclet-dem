// Correctness of Kokkos periodic ghost generation (dem::generateGhostsKokkos). Real particles, some
// near periodic faces; generate ghosts and verify: (1) total ghost count == host expectation, and
// (2) each ghost is its real owner's state shifted by an integer multiple of the box on each axis
// (positions/posPred shifted; velocities/quat/scale/shape copied verbatim; real_indices correct).
// Ghost slot ordering is nondeterministic (atomic append), so we validate per-ghost, not by slot.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "periodicity.hpp"

using namespace dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int numReal = 600;
    const int capacity = numReal * 8;
    const float skin = 0.5f;
    Domain dom{F3{-5,-5,-5}, F3{5,5,5}, F3{10,10,10}, true, true, true};

    std::mt19937 rng(23);
    std::uniform_real_distribution<float> uf(-5.f, 5.f);
    std::uniform_real_distribution<float> uedge(0.f, 1.f);
    std::normal_distribution<float> nq(0.f, 1.f);

    std::vector<float> px(capacity),py(capacity),pz(capacity);
    std::vector<float> vx(capacity),vy(capacity),vz(capacity);
    std::vector<float> qx(capacity),qy(capacity),qz(capacity),qw(capacity), sc(capacity), im(capacity);
    std::vector<int> sid(capacity);
    for (int i=0;i<numReal;++i){
      // bias ~half the particles near a face to exercise multi-axis ghosts
      auto coord=[&](){ float r=uedge(rng); if(r<0.25f) return -5.f+uedge(rng)*0.4f; if(r<0.5f) return 5.f-uedge(rng)*0.4f; return uf(rng); };
      px[i]=coord(); py[i]=coord(); pz[i]=coord();
      vx[i]=uf(rng);vy[i]=uf(rng);vz[i]=uf(rng);
      float a=nq(rng),b=nq(rng),c=nq(rng),d=nq(rng); float nl=std::sqrt(a*a+b*b+c*c+d*d)+1e-12f;
      qx[i]=a/nl;qy[i]=b/nl;qz[i]=c/nl;qw[i]=d/nl;
      sc[i]=0.5f+uedge(rng); im[i]=1.f; sid[i]=i%4;
    }

    auto mk3=[&](const char*n,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z){
      V3 v(n,capacity); auto h=Kokkos::create_mirror_view(v); for(int i=0;i<capacity;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];} Kokkos::deep_copy(v,h); return v;};
    auto mk4=[&](const char*n,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z,std::vector<float>&w){
      V4 v(n,capacity); auto h=Kokkos::create_mirror_view(v); for(int i=0;i<capacity;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];h(i,3)=w[i];} Kokkos::deep_copy(v,h); return v;};
    auto pos=mk3("pos",px,py,pz), posPred=mk3("pp",px,py,pz), vel=mk3("vel",vx,vy,vz), velPred=mk3("vp",vx,vy,vz);
    auto angVel=mk3("av",vx,vy,vz), angVelPred=mk3("avp",vx,vy,vz);
    auto quat=mk4("q",qx,qy,qz,qw), quatPred=mk4("qp",qx,qy,qz,qw);
    Vf invMass("im",capacity), scale("sc",capacity);
    { auto h=Kokkos::create_mirror_view(invMass); for(int i=0;i<capacity;++i)h(i)=im[i]; Kokkos::deep_copy(invMass,h);}
    { auto h=Kokkos::create_mirror_view(scale); for(int i=0;i<capacity;++i)h(i)=sc[i]; Kokkos::deep_copy(scale,h);}
    Vi shapeId("sid",capacity), realIdx("ri",capacity);
    { auto h=Kokkos::create_mirror_view(shapeId); for(int i=0;i<capacity;++i)h(i)=sid[i]; Kokkos::deep_copy(shapeId,h);}
    Kokkos::View<int,CpMem> top("top"); Kokkos::deep_copy(top, numReal);

    generateGhostsKokkos(numReal, capacity, dom, skin, pos, invMass, posPred, vel, velPred, quat,
                         quatPred, angVel, angVelPred, scale, shapeId, realIdx, top);

    int total=0; Kokkos::deep_copy(total, top);

    // host expected ghost count
    int expected=0;
    for (int i=0;i<numReal;++i){
      int sx = (px[i]<dom.min.x+skin)?1:((px[i]>dom.max.x-skin)?-1:0);
      int sy = (py[i]<dom.min.y+skin)?1:((py[i]>dom.max.y-skin)?-1:0);
      int sz = (pz[i]<dom.min.z+skin)?1:((pz[i]>dom.max.z-skin)?-1:0);
      int nx=(sx==0)?1:2, ny=(sy==0)?1:2, nz=(sz==0)?1:2;
      expected += nx*ny*nz - 1;
    }
    if (total - numReal != expected) {
      std::fprintf(stderr, "FAIL: ghost count %d != expected %d\n", total-numReal, expected);
      status = 1;
    }

    // validate each ghost
    auto h3=[&](V3 v){ auto h=Kokkos::create_mirror_view(v); Kokkos::deep_copy(h,v); return h; };
    auto h4=[&](V4 v){ auto h=Kokkos::create_mirror_view(v); Kokkos::deep_copy(h,v); return h; };
    auto hpos=h3(pos), hpp=h3(posPred), hvel=h3(vel); auto hq=h4(quat);
    auto hri=Kokkos::create_mirror_view(realIdx); Kokkos::deep_copy(hri,realIdx);
    auto hsc=Kokkos::create_mirror_view(scale); Kokkos::deep_copy(hsc,scale);
    auto hsid=Kokkos::create_mirror_view(shapeId); Kokkos::deep_copy(hsid,shapeId);
    auto close=[&](float a,float b){return std::fabs(a-b)<=1e-3f;};
    int bad=0;
    for (int g=numReal; g<total; ++g){
      int r=hri(g);
      if (r<0||r>=numReal){ ++bad; continue; }
      // shift must be integer multiple of size on each axis, in {-1,0,1}
      float kx=(hpos(g,0)-px[r])/dom.size.x, ky=(hpos(g,1)-py[r])/dom.size.y, kz=(hpos(g,2)-pz[r])/dom.size.z;
      auto okk=[&](float k){ float rk=std::round(k); return std::fabs(k-rk)<1e-3f && std::fabs(rk)<=1.0001f; };
      if(!okk(kx)||!okk(ky)||!okk(kz)){ ++bad; continue; }
      if(std::round(kx)==0&&std::round(ky)==0&&std::round(kz)==0){ ++bad; continue; }  // not a self-copy
      // posPred shifted by same offset
      if(!close(hpp(g,0), px[r]+std::round(kx)*dom.size.x) || !close(hpp(g,1), py[r]+std::round(ky)*dom.size.y) || !close(hpp(g,2), pz[r]+std::round(kz)*dom.size.z)){ ++bad; continue; }
      // verbatim copies
      if(!close(hvel(g,0),vx[r])||!close(hvel(g,1),vy[r])||!close(hvel(g,2),vz[r]))++bad;
      if(!close(hq(g,0),qx[r])||!close(hq(g,3),qw[r]))++bad;
      if(!close(hsc(g),sc[r])||hsid(g)!=sid[r])++bad;
    }
    if (bad){ std::fprintf(stderr, "FAIL: %d ghosts invalid\n", bad); status=1; }
    if(!status) std::printf("[periodicity] PASS: %d ghosts from %d reals correct (exec: %s)\n", total-numReal, numReal, CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
