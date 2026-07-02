// Correctness of the Kokkos friction cluster (compute_plane_load -> accumulate_normal_impulse ->
// count_friction_contacts -> solve_contact_friction) against a host replication of the identical
// math, run as the same 4-step sequence. Candidate contacts with a borderline normal approach are
// dropped so the active/inactive decision (and the integer per-body counts) are decisive on both
// host and device. Compares friction_lambda_n, plane-load/count, and the friction delta_vel/
// delta_ang_vel. Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "solver_friction.hpp"

using namespace peclet::dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 400;
    const float growthRate = 0.01f;
    const float frictionDynamic = 0.5f;

    std::mt19937 rng(53);
    std::uniform_real_distribution<float> uf(-1.f, 1.f);
    std::uniform_real_distribution<float> upos(0.5f, 2.0f);
    std::uniform_int_distribution<int> ubody(0, N - 1);

    std::vector<float> invMass(N), iIx(N), iIy(N), iIz(N);
    std::vector<float> vx(N), vy(N), vz(N), wx(N), wy(N), wz(N);
    for (int i = 0; i < N; ++i) {
      invMass[i]=upos(rng); iIx[i]=upos(rng); iIy[i]=upos(rng); iIz[i]=upos(rng);
      vx[i]=uf(rng); vy[i]=uf(rng); vz[i]=uf(rng); wx[i]=uf(rng); wy[i]=uf(rng); wz[i]=uf(rng);
    }
    auto vP=[&](int i){return F3{vx[i],vy[i],vz[i]};};
    auto wP=[&](int i){return F3{wx[i],wy[i],wz[i]};};
    auto iI=[&](int i){return F3{iIx[i],iIy[i],iIz[i]};};

    // Build a contact list dropping borderline normal approach (|approach| < 0.05).
    std::vector<ContactC> contacts;
    for (int k = 0; k < 4000; ++k) {
      ContactC c{};
      int a=ubody(rng), b;
      if ((rng()&3)==0) b=-1; else { do{b=ubody(rng);}while(b==a); if(b<a){int t=a;a=b;b=t;} }
      c.bodyA=a; c.bodyB=b;
      float nx=uf(rng),ny=uf(rng),nz=uf(rng); float nl=std::sqrt(nx*nx+ny*ny+nz*nz)+1e-12f;
      c.normal=F4{nx/nl,ny/nl,nz/nl,0};
      c.rA=F4{uf(rng)*0.5f,uf(rng)*0.5f,uf(rng)*0.5f,0};
      c.rB=F4{uf(rng)*0.5f,uf(rng)*0.5f,uf(rng)*0.5f,0};
      c.dist=0; c.friction_lambda_n=0; c.weight=0;
      F3 rA{c.rA.x,c.rA.y,c.rA.z}, rB{c.rB.x,c.rB.y,c.rB.z}, n{c.normal.x,c.normal.y,c.normal.z};
      float approach;
      if (b<0) { F3 vAc=add3(vP(a),cross3v(wP(a),rA)); approach=-dot3(vAc,n); }
      else { F3 vAc=add3(vP(a),cross3v(wP(a),rA)); F3 vBc=add3(vP(b),cross3v(wP(b),rB));
             F3 vrel=add3(sub3(vAc,vBc),scale3(sub3(rA,rB),growthRate)); approach=-dot3(vrel,n); }
      if (std::fabs(approach) < 0.05f) continue;
      contacts.push_back(c);
    }
    const int M=(int)contacts.size();

    // --- upload ---
    Kokkos::View<ContactC*, CpMem> dC("c", M);
    auto hC=Kokkos::create_mirror_view(dC); for(int k=0;k<M;++k)hC(k)=contacts[k]; Kokkos::deep_copy(dC,hC);
    Kokkos::View<float*, CpMem> dIM("im",N);
    { auto h=Kokkos::create_mirror_view(dIM); for(int i=0;i<N;++i)h(i)=invMass[i]; Kokkos::deep_copy(dIM,h);}
    auto up3=[&](const char*nm,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z){
      Kokkos::View<float*[3],CpMem> v(nm,N); auto h=Kokkos::create_mirror_view(v);
      for(int i=0;i<N;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];} Kokkos::deep_copy(v,h); return v;};
    auto dInvI=up3("ii",iIx,iIy,iIz);
    auto dVel=up3("v",vx,vy,vz);
    auto dAng=up3("w",wx,wy,wz);
    Kokkos::View<int*,CpMem> dReal("r",N);
    { auto h=Kokkos::create_mirror_view(dReal); for(int i=0;i<N;++i)h(i)=i; Kokkos::deep_copy(dReal,h);}
    FrManifoldCounts dPF("pf",N);
    Kokkos::View<float*[3],CpMem> dDV("dv",N), dDW("dw",N);

    // --- device sequence ---
    computePlaneLoadKokkos(dC, M, dIM, dInvI, dVel, dAng, dPF);
    accumulateNormalImpulseKokkos(dC, M, dIM, dInvI, dVel, dAng, dReal, growthRate);
    countFrictionContactsKokkos(dC, M, dReal, dPF);
    solveContactFrictionKokkos(dC, M, dIM, dInvI, dVel, dAng, dReal, dPF, frictionDynamic, dDV, dDW);

    std::vector<float> gln(M), gpx(N), gpy(N), gdv(3*N), gdw(3*N);
    { auto h=Kokkos::create_mirror_view(dC); Kokkos::deep_copy(h,dC); for(int k=0;k<M;++k)gln[k]=h(k).friction_lambda_n; }
    { auto h=Kokkos::create_mirror_view(dPF); Kokkos::deep_copy(h,dPF); for(int i=0;i<N;++i){gpx[i]=h(i,0);gpy[i]=h(i,1);} }
    { auto h=Kokkos::create_mirror_view(dDV); Kokkos::deep_copy(h,dDV); for(int i=0;i<N;++i){gdv[3*i]=h(i,0);gdv[3*i+1]=h(i,1);gdv[3*i+2]=h(i,2);} }
    { auto h=Kokkos::create_mirror_view(dDW); Kokkos::deep_copy(h,dDW); for(int i=0;i<N;++i){gdw[3*i]=h(i,0);gdw[3*i+1]=h(i,1);gdw[3*i+2]=h(i,2);} }

    // --- host replication of the same 4 steps ---
    std::vector<float> ln(M,0), pfx(N,0), pfy(N,0), rdv(3*N,0), rdw(3*N,0);
    auto cW=[&](F3 r,F3 dir,float invM,F3 invI){ F3 rn=cross3v(r,dir);
      return invM+rn.x*rn.x*invI.x+rn.y*rn.y*invI.y+rn.z*rn.z*invI.z; };
    // 1. plane load
    for (int k=0;k<M;++k){ const ContactC&c=contacts[k]; if(c.bodyB>=0)continue; int a=c.bodyA;
      if(invMass[a]<=0)continue; F3 rA{c.rA.x,c.rA.y,c.rA.z}, n{c.normal.x,c.normal.y,c.normal.z};
      F3 vAc=add3(vP(a),cross3v(wP(a),rA)); float ap=-dot3(vAc,n); if(ap<=0)continue;
      float wn=cW(rA,n,invMass[a],iI(a)); ln[k]=(wn>1e-6f)?ap/wn:0.f;
      float load=ap/invMass[a]; if(load>pfx[a])pfx[a]=load; }
    // 2. accumulate body-body
    for (int k=0;k<M;++k){ const ContactC&c=contacts[k]; if(c.bodyB<0)continue; int a=c.bodyA,b=c.bodyB;
      if(a>b)continue; F3 rA{c.rA.x,c.rA.y,c.rA.z}, rB{c.rB.x,c.rB.y,c.rB.z}, n{c.normal.x,c.normal.y,c.normal.z};
      F3 vAc=add3(vP(a),cross3v(wP(a),rA)); F3 vBc=add3(vP(b),cross3v(wP(b),rB));
      F3 vrel=add3(sub3(vAc,vBc),scale3(sub3(rA,rB),growthRate)); float ap=-dot3(vrel,n); if(ap<=0)continue;
      float wn=cW(rA,n,invMass[a],iI(a))+cW(rB,n,invMass[b],iI(b)); if(wn>1e-6f)ln[k]+=ap/wn; }
    // 3. count
    for (int k=0;k<M;++k){ if(ln[k]<=0)continue; pfy[contacts[k].bodyA]+=1.f; if(contacts[k].bodyB>=0)pfy[contacts[k].bodyB]+=1.f; }
    // 4. solve
    for (int k=0;k<M;++k){ const ContactC&c=contacts[k]; float lam=ln[k]; if(lam<=0)continue;
      int a=c.bodyA,b=c.bodyB; float invMA=invMass[a], invMB=(b>=0)?invMass[b]:0.f;
      F3 invIA=iI(a), invIB=(b>=0)?iI(b):F3{0,0,0};
      F3 rA{c.rA.x,c.rA.y,c.rA.z}, rB{c.rB.x,c.rB.y,c.rB.z}, n{c.normal.x,c.normal.y,c.normal.z};
      F3 vAc=add3(vP(a),cross3v(wP(a),rA)); F3 vBc{0,0,0}; if(b>=0)vBc=add3(vP(b),cross3v(wP(b),rB));
      F3 vrel=sub3(vAc,vBc); float vn=dot3(vrel,n); F3 vt=sub3(vrel,scale3(n,vn)); float vl=std::sqrt(dot3(vt,vt));
      if(vl<1e-7f)continue; F3 t=scale3(vt,1.f/vl);
      F3 rnA=cross3v(rA,t), rnB=cross3v(rB,t);
      float wt=invMA+invMB+rnA.x*rnA.x*invIA.x+rnA.y*rnA.y*invIA.y+rnA.z*rnA.z*invIA.z
               +rnB.x*rnB.x*invIB.x+rnB.y*rnB.y*invIB.y+rnB.z*rnB.z*invIB.z; if(wt<1e-6f)continue;
      float bound=(b<0)?pfx[a]:lam; float nA=pfy[a], nB=(b>=0)?pfy[b]:0.f;
      float invn=1.f/std::fmax(std::fmax(nA,nB),1.f);
      float lt=-vl/wt; float mf=frictionDynamic*bound; if(lt<-mf)lt=-mf; lt*=invn;
      rdv[3*a]+=t.x*lt*invMA; rdv[3*a+1]+=t.y*lt*invMA; rdv[3*a+2]+=t.z*lt*invMA;
      rdw[3*a]+=rnA.x*invIA.x*lt; rdw[3*a+1]+=rnA.y*invIA.y*lt; rdw[3*a+2]+=rnA.z*invIA.z*lt;
      if(b>=0){ rdv[3*b]+=-t.x*lt*invMB; rdv[3*b+1]+=-t.y*lt*invMB; rdv[3*b+2]+=-t.z*lt*invMB;
        rdw[3*b]+=-rnB.x*invIB.x*lt; rdw[3*b+1]+=-rnB.y*invIB.y*lt; rdw[3*b+2]+=-rnB.z*invIB.z*lt; } }

    int bad=0;
    auto tol=[&](float g,float r){ return std::fabs(g-r)<=1e-4f*(1+std::fabs(r)); };
    for(int k=0;k<M;++k) if(!tol(gln[k],ln[k])) ++bad;
    for(int i=0;i<N;++i){ if(!tol(gpx[i],pfx[i]))++bad; if(gpy[i]!=pfy[i])++bad; }
    for(int i=0;i<3*N;++i){ if(!tol(gdv[i],rdv[i]))++bad; if(!tol(gdw[i],rdw[i]))++bad; }
    if (bad) { std::fprintf(stderr, "FAIL: %d friction quantities differ\n", bad); status=1; }
    else std::printf("[solver_friction] PASS: %d contacts, lambda_n/plane-load/count/deltas match host (exec: %s)\n", M, CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
