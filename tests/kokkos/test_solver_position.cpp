// Correctness of the Kokkos XPBD position solve (dem::solvePositionKokkos) against a host replication
// of the identical math. Random bodies + contacts; candidate contacts with a borderline constraint
// value |C| are dropped so the active/inactive decision is decisive on both host and device (keeps
// constraint_counts exactly comparable). Compares accumulated delta_pos / delta_quat (within tol),
// constraint_counts (exact), and max_overlap. Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "solver_position.hpp"

using namespace dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 400;
    std::mt19937 rng(31);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_real_distribution<float> upos(0.5f, 2.0f);
    std::uniform_int_distribution<int> ubody(0, N - 1);
    std::normal_distribution<float> nq(0.f, 1.f);

    std::vector<float> invMass(N), iIx(N), iIy(N), iIz(N);
    std::vector<float> qpx(N), qpy(N), qpz(N), qpw(N), qsx(N), qsy(N), qsz(N), qsw(N);
    std::vector<float> px(N), py(N), pz(N);
    auto randq = [&](std::vector<float>& x, std::vector<float>& y, std::vector<float>& z,
                     std::vector<float>& w, int i) {
      float a=nq(rng),b=nq(rng),c=nq(rng),d=nq(rng); float nrm=std::sqrt(a*a+b*b+c*c+d*d)+1e-12f;
      x[i]=a/nrm; y[i]=b/nrm; z[i]=c/nrm; w[i]=d/nrm;
    };
    for (int i = 0; i < N; ++i) {
      invMass[i]=upos(rng); iIx[i]=upos(rng); iIy[i]=upos(rng); iIz[i]=upos(rng);
      px[i]=uf(rng)*5; py[i]=uf(rng)*5; pz[i]=uf(rng)*5;
      randq(qpx,qpy,qpz,qpw,i); randq(qsx,qsy,qsz,qsw,i);
    }
    auto qp=[&](int i){return F4{qpx[i],qpy[i],qpz[i],qpw[i]};};
    auto qs=[&](int i){return F4{qsx[i],qsy[i],qsz[i],qsw[i]};};
    auto pp=[&](int i){return F3{px[i],py[i],pz[i]};};

    // Compute the constraint value C of a contact exactly as the kernel does (for filtering + ref).
    auto computeC = [&](const ContactC& c, F3& rA_out, F3& rB_out, F3& n_out) {
      int idA=c.bodyA, idB=c.bodyB;
      F3 pA=pp(idA); F4 qA=qp(idA);
      F4 qAd=quatMult(qA,quatInverse(qs(idA)));
      F3 rA=rotateVector(qAd,F3{c.rA.x,c.rA.y,c.rA.z});
      F3 rB{c.rB.x,c.rB.y,c.rB.z}; F3 n{c.normal.x,c.normal.y,c.normal.z};
      F3 pB{0,0,0};
      if(idB>=0){ pB=pp(idB); F4 qB=qp(idB); F4 qBd=quatMult(qB,quatInverse(qs(idB))); rB=rotateVector(qBd,rB); n=rotateVector(qBd,n); }
      float C;
      if(idB<0){ n=F3{c.normal.x,c.normal.y,c.normal.z}; C=dot3(sub3(add3(pA,rA),F3{c.rB.x,c.rB.y,c.rB.z}),n); }
      else { C=dot3(sub3(add3(pA,rA),add3(pB,rB)),n); }
      rA_out=rA; rB_out=rB; n_out=n; return C;
    };

    // Build a contact list, dropping borderline |C| < 0.05.
    std::vector<ContactC> contacts;
    for (int k = 0; k < 4000; ++k) {
      ContactC c{};
      int a=ubody(rng), b;
      if ((rng()&7)==0) b=-1; else { do{b=ubody(rng);}while(b==a); }
      c.bodyA=a; c.bodyB=b;
      float nx=uf(rng),ny=uf(rng),nz=uf(rng); float nl=std::sqrt(nx*nx+ny*ny+nz*nz)+1e-12f;
      c.normal=F4{nx/nl,ny/nl,nz/nl,0};
      c.rA=F4{uf(rng)*0.5f,uf(rng)*0.5f,uf(rng)*0.5f,0};
      c.rB=F4{uf(rng)*0.5f,uf(rng)*0.5f,uf(rng)*0.5f,0};
      c.dist=0; c.friction_lambda_n=0; c.weight=0;
      F3 ra,rb,n; float C=computeC(c,ra,rb,n);
      if (std::fabs(C) < 0.05f) continue;  // drop borderline
      contacts.push_back(c);
    }
    const int M = (int)contacts.size();

    // --- upload ---
    Kokkos::View<ContactC*, CpMem> dC("c", M);
    { auto h=Kokkos::create_mirror_view(dC); for(int k=0;k<M;++k)h(k)=contacts[k]; Kokkos::deep_copy(dC,h);}
    Kokkos::View<float*, CpMem> dIM("im", N);
    { auto h=Kokkos::create_mirror_view(dIM); for(int i=0;i<N;++i)h(i)=invMass[i]; Kokkos::deep_copy(dIM,h);}
    auto up3=[&](const char*nm,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z){
      Kokkos::View<float*[3],CpMem> v(nm,N); auto h=Kokkos::create_mirror_view(v);
      for(int i=0;i<N;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];} Kokkos::deep_copy(v,h); return v;};
    auto up4=[&](const char*nm,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z,std::vector<float>&w){
      Kokkos::View<float*[4],CpMem> v(nm,N); auto h=Kokkos::create_mirror_view(v);
      for(int i=0;i<N;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];h(i,3)=w[i];} Kokkos::deep_copy(v,h); return v;};
    auto dPos=up3("pos",px,py,pz);
    auto dQp=up4("qp",qpx,qpy,qpz,qpw);
    auto dQs=up4("qs",qsx,qsy,qsz,qsw);
    auto dInvI=up3("ii",iIx,iIy,iIz);
    Kokkos::View<float*[3],CpMem> dDpos("dpos",N);
    Kokkos::View<float*[4],CpMem> dDquat("dquat",N);
    Kokkos::View<int*,CpMem> dCount("cnt",N);
    Kokkos::View<float,CpMem> dMaxOv("mo");

    solvePositionKokkos(dC, M, dIM, dPos, dQp, dQs, dInvI, dDpos, dDquat, dCount, dMaxOv);

    std::vector<float> gdp(3*N), gdq(4*N); std::vector<int> gcn(N); float gmo=0;
    { auto h=Kokkos::create_mirror_view(dDpos); Kokkos::deep_copy(h,dDpos); for(int i=0;i<N;++i){gdp[3*i]=h(i,0);gdp[3*i+1]=h(i,1);gdp[3*i+2]=h(i,2);} }
    { auto h=Kokkos::create_mirror_view(dDquat); Kokkos::deep_copy(h,dDquat); for(int i=0;i<N;++i){gdq[4*i]=h(i,0);gdq[4*i+1]=h(i,1);gdq[4*i+2]=h(i,2);gdq[4*i+3]=h(i,3);} }
    { auto h=Kokkos::create_mirror_view(dCount); Kokkos::deep_copy(h,dCount); for(int i=0;i<N;++i)gcn[i]=h(i); }
    { auto h=Kokkos::create_mirror_view(dMaxOv); Kokkos::deep_copy(h,dMaxOv); gmo=h(); }

    // --- host reference ---
    std::vector<float> rdp(3*N,0), rdq(4*N,0); std::vector<int> rcn(N,0); float rmo=0;
    for (const auto& c : contacts) {
      int idA=c.bodyA, idB=c.bodyB;
      F3 rA,rB,n; float C=computeC(c,rA,rB,n);
      if (C>=0.f) continue;
      float invMA=invMass[idA], invMB=(idB>=0)?invMass[idB]:0.f;
      F3 invIA{iIx[idA],iIy[idA],iIz[idA]};
      F3 invIB=(idB>=0)?F3{iIx[idB],iIy[idB],iIz[idB]}:F3{0,0,0};
      float wT=detail::computeW(rA,n,invMA,invIA)+detail::computeW(rB,n,invMB,invIB);
      if (wT<1e-6f) continue;
      float dL=-C/wT;
      rdp[3*idA]+=n.x*dL*invMA; rdp[3*idA+1]+=n.y*dL*invMA; rdp[3*idA+2]+=n.z*dL*invMA;
      { F3 rn=cross3v(rA,n); F3 dT{rn.x*invIA.x*dL,rn.y*invIA.y*dL,rn.z*invIA.z*dL}; F4 dq=detail::deltaQuat(dT,qp(idA));
        rdq[4*idA]+=dq.x;rdq[4*idA+1]+=dq.y;rdq[4*idA+2]+=dq.z;rdq[4*idA+3]+=dq.w; }
      if(idB>=0){
        rdp[3*idB]+=-n.x*dL*invMB; rdp[3*idB+1]+=-n.y*dL*invMB; rdp[3*idB+2]+=-n.z*dL*invMB;
        F3 rn=cross3v(rB,n); F3 dT{-rn.x*invIB.x*dL,-rn.y*invIB.y*dL,-rn.z*invIB.z*dL}; F4 dq=detail::deltaQuat(dT,qp(idB));
        rdq[4*idB]+=dq.x;rdq[4*idB+1]+=dq.y;rdq[4*idB+2]+=dq.z;rdq[4*idB+3]+=dq.w;
        rcn[idB]++;
      }
      rcn[idA]++;
      if (C<0.f && -C>rmo) rmo=-C;
    }

    int bad=0, badc=0;
    for (int i=0;i<3*N;++i) if (std::fabs(gdp[i]-rdp[i])>1e-4f*(1+std::fabs(rdp[i]))) ++bad;
    for (int i=0;i<4*N;++i) if (std::fabs(gdq[i]-rdq[i])>1e-4f*(1+std::fabs(rdq[i]))) ++bad;
    for (int i=0;i<N;++i) if (gcn[i]!=rcn[i]) ++badc;
    bool moOk = std::fabs(gmo-rmo) <= 1e-4f*(1+rmo);
    if (bad||badc||!moOk) {
      std::fprintf(stderr, "FAIL: delta diffs=%d, count diffs=%d, maxov dev=%.3e host=%.3e\n", bad, badc, gmo, rmo);
      status=1;
    } else {
      std::printf("[solver_position] PASS: %d contacts, deltas/counts/max_overlap match host (exec: %s)\n",
                  M, CpExec::name());
    }
  }
  Kokkos::finalize();
  return status;
}
