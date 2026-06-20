// Correctness of the Kokkos Berendsen thermostat (dem::applyThermostatKokkos) against a host
// replication: reduce translational + rotational KE, compute the two scaling factors, rescale
// velocities. Device reduction order differs from the host sequential sum, so compare the rescaled
// velocities within tolerance. Runs on whatever backend Kokkos was built for.
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "integration.hpp"

using namespace dem;

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    const int N = 1000;
    const double kB = 1.0, tau = 0.1, Ttarget = 2.0;
    const float dt = 0.01f;

    std::mt19937 rng(71);
    std::uniform_real_distribution<float> uf(-1.5f, 1.5f);
    std::uniform_real_distribution<float> up(0.5f, 2.0f);
    std::normal_distribution<float> nq(0.f, 1.f);
    std::vector<float> vx(N),vy(N),vz(N),wx(N),wy(N),wz(N),im(N),iix(N),iiy(N),iiz(N),qx(N),qy(N),qz(N),qw(N);
    for (int i=0;i<N;++i){
      vx[i]=uf(rng);vy[i]=uf(rng);vz[i]=uf(rng);wx[i]=uf(rng);wy[i]=uf(rng);wz[i]=uf(rng);
      im[i]=up(rng); iix[i]=up(rng);iiy[i]=up(rng);iiz[i]=up(rng);
      float a=nq(rng),b=nq(rng),c=nq(rng),d=nq(rng); float nl=std::sqrt(a*a+b*b+c*c+d*d)+1e-12f;
      qx[i]=a/nl;qy[i]=b/nl;qz[i]=c/nl;qw[i]=d/nl;
    }
    auto mk3=[&](const char*n,std::vector<float>&x,std::vector<float>&y,std::vector<float>&z){
      V3 v(n,N); auto h=Kokkos::create_mirror_view(v); for(int i=0;i<N;++i){h(i,0)=x[i];h(i,1)=y[i];h(i,2)=z[i];} Kokkos::deep_copy(v,h); return v;};
    auto vel=mk3("v",vx,vy,vz), ang=mk3("w",wx,wy,wz), invI=mk3("ii",iix,iiy,iiz);
    V4 quat("q",N); { auto h=Kokkos::create_mirror_view(quat); for(int i=0;i<N;++i){h(i,0)=qx[i];h(i,1)=qy[i];h(i,2)=qz[i];h(i,3)=qw[i];} Kokkos::deep_copy(quat,h);}
    Vf invMass("im",N); { auto h=Kokkos::create_mirror_view(invMass); for(int i=0;i<N;++i)h(i)=im[i]; Kokkos::deep_copy(invMass,h);}

    applyThermostatKokkos(N, vel, invMass, ang, invI, quat, kB, tau, Ttarget, dt);

    std::vector<float> gv(3*N), gw(3*N);
    { auto h=Kokkos::create_mirror_view(vel); Kokkos::deep_copy(h,vel); for(int i=0;i<N;++i){gv[3*i]=h(i,0);gv[3*i+1]=h(i,1);gv[3*i+2]=h(i,2);} }
    { auto h=Kokkos::create_mirror_view(ang); Kokkos::deep_copy(h,ang); for(int i=0;i<N;++i){gw[3*i]=h(i,0);gw[3*i+1]=h(i,1);gw[3*i+2]=h(i,2);} }

    // host reference
    double KEt=0, KEr=0;
    for (int i=0;i<N;++i){
      KEt += 0.5*(1.0/im[i])*(vx[i]*vx[i]+vy[i]*vy[i]+vz[i]*vz[i]);
      F3 wb=invRotateVector(F4{qx[i],qy[i],qz[i],qw[i]}, F3{wx[i],wy[i],wz[i]});
      double Ix=1.0/iix[i],Iy=1.0/iiy[i],Iz=1.0/iiz[i];
      KEr += 0.5*(Ix*wb.x*wb.x+Iy*wb.y*wb.y+Iz*wb.z*wb.z);
    }
    double ndof=3.0*N;
    auto lam=[&](double ke){ double Tc=(2.0*ke)/(ndof*kB); if(Tc<1e-6)return 1.0; double f=1.0+(dt/tau)*(Ttarget/Tc-1.0); if(f<0)f=0; return std::sqrt(f); };
    float lt=(float)lam(KEt), lr=(float)lam(KEr);

    int bad=0; auto tol=[&](float a,float b){return std::fabs(a-b)<=1e-4f*(1+std::fabs(b));};
    for (int i=0;i<N;++i){
      if(!tol(gv[3*i],vx[i]*lt)||!tol(gv[3*i+1],vy[i]*lt)||!tol(gv[3*i+2],vz[i]*lt))++bad;
      if(!tol(gw[3*i],wx[i]*lr)||!tol(gw[3*i+1],wy[i]*lr)||!tol(gw[3*i+2],wz[i]*lr))++bad;
    }
    if(bad){ std::fprintf(stderr,"FAIL: %d/%d scaled velocities differ (lt=%f lr=%f)\n",bad,6*N,lt,lr); status=1; }
    else std::printf("[thermostat] PASS: %d bodies rescaled (lt=%.4f lr=%.4f) match host (exec: %s)\n",N,lt,lr,CpExec::name());
  }
  Kokkos::finalize();
  return status;
}
