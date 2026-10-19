#ifndef MHD_CHIRAL_DYNAMO_HPP_
#define MHD_CHIRAL_DYNAMO_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chiral_dynamo.hpp
//! \brief solver-agnostic chiral magnetic effect (CME) physics
//!
//! The chiral dynamo couples a chiral-imbalance scalar Y5 to the magnetic field
//! through a modified Ohm's law, e^mu = xi * b^mu.  The same four pieces of
//! physics are needed by every solver that implements it, so they live here
//! rather than being duplicated per solver:
//!
//!   Xi()            dynamo coefficient xi(Y5, Ye)
//!   OhmsLawEMF()     the 3+1 electric field E_i that follows from e^mu = xi b^mu
//!   GammaM()         chirality-flip rate, in code units
//!   AnomalySource()  the E.B chiral anomaly source for dn5/dt
//!
//! The closed form used by OhmsLawEMF() is derived and checked against a direct
//! linear solve in scripts/ohms_law_dynamo.py of the ChiralDynamo superproject,
//! which also verifies the ideal-MHD (xi -> 0) and comoving (v -> 0) limits.
//! Re-run that script if the algebra here is touched.
//!
//! The chiral EMF enters only the induction equation, as a perturbation of ideal
//! GRMHD.  The fluxes and C2P assume e^mu = 0, so the stress-energy is wrong
//! at O(xi) (exact: (1+xi^2) times the ideal form, with b^mu from B and E); a
//! consistent treatment needs a beyond-ideal-MHD formulation.
//!
//! Current users: mhd_corner_e.cpp (2D and 3D EMF) and
//! radiation_m1_update.cpp (Gamma_m + E.B kernel).

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace chiral {

//! Fine-structure constant.  This used to be 1/137.0 in mhd_corner_e.cpp and
//! 1/137.036 in radiation_m1_update.cpp; both now use this value so that the
//! EMF and the anomaly source are built from the same coupling.
constexpr Real kAlphaEM = 1.0/137.035999;

//! Electron mass, MeV
constexpr Real kElectronMassMeV = 0.510999;

//! Reduced Planck constant, MeV s
constexpr Real kHbarMeVs = 6.582119569e-22;

//! Reduced Planck constant, erg s
constexpr Real kHbarErgS = 1.054571817e-27;

//----------------------------------------------------------------------------------------
//! \fn Real chiral::XiCoeff
//! \brief prefactor of the dynamo coefficient, -(4/pi) alpha^2 ln(1/alpha)
//!
//! Split out from Xi() only so callers that need the bare coefficient (e.g. for
//! a diagnostic) do not have to repeat it.
KOKKOS_INLINE_FUNCTION
Real XiCoeff(const Real alpha_em) {
  return -(4.0/M_PI) * SQR(alpha_em) * Kokkos::log(1.0/alpha_em);
}

//----------------------------------------------------------------------------------------
//! \fn Real chiral::Xi
//! \brief dynamo coefficient xi = -(4/pi) alpha^2 ln(1/alpha) (Y5/Ye)^(1/3)
//!
//! Returns 0 for Ye <= 0, which is not a physical state but can be produced
//! transiently by prolongation or a failed primitive solve.
KOKKOS_INLINE_FUNCTION
Real Xi(const Real Y5, const Real Ye, const Real alpha_em) {
  return (Ye > 0.0) ? XiCoeff(alpha_em) * Kokkos::cbrt(Y5/Ye) : 0.0;
}

//----------------------------------------------------------------------------------------
//! \fn void chiral::OhmsLawEMF
//! \brief 3+1 electric field from the dynamo Ohm's law e^mu = xi b^mu
//!
//! Inputs are the Valencia three-velocity v^i (W v^i is what the DynGR solver
//! stores in IVX..IVZ), the densitized field sqrt(gamma) B^i, 1/W, and the
//! spatial metric gamma_ij with 1/sqrt(gamma).  Returns the covariant Eulerian
//! electric field E_i.  v^2 = gamma_ij v^i v^j = 1 - 1/W^2, so the Lorentz factor
//! enters only through iW.
//!
//! The expression is the flat-space closed form of scripts/ohms_law_dynamo.py
//! written with gamma_ij and eps_ijk = sqrt(gamma) [ijk]; it holds in any metric
//! because it is a 3-tensor identity in the Eulerian observer's frame.
//!
//! xi -> 0 recovers the ideal-MHD EMF, E = -v x B.
KOKKOS_INLINE_FUNCTION
void OhmsLawEMF(const Real xi, const Real iW,
                const Real v1, const Real v2, const Real v3,
                const Real b1, const Real b2, const Real b3,
                const Real gxx, const Real gxy, const Real gxz,
                const Real gyy, const Real gyz, const Real gzz,
                const Real isqrtg,
                Real &e1, Real &e2, Real &e3) {
  const Real v_1 = gxx*v1 + gxy*v2 + gxz*v3;
  const Real v_2 = gxy*v1 + gyy*v2 + gyz*v3;
  const Real v_3 = gxz*v1 + gyz*v2 + gzz*v3;
  const Real B_1 = (gxx*b1 + gxy*b2 + gxz*b3)*isqrtg;
  const Real B_2 = (gxy*b1 + gyy*b2 + gyz*b3)*isqrtg;
  const Real B_3 = (gxz*b1 + gyz*b2 + gzz*b3)*isqrtg;

  const Real v2sq  = 1.0 - SQR(iW);
  const Real xi2   = SQR(xi);
  const Real D     = 1.0/(1.0 + xi2*v2sq);
  const Real vdotB = (v_1*b1 + v_2*b2 + v_3*b3)*isqrtg;

  e1 = D*(-(1+xi2)*(v2*b3 - v3*b2) + xi*(1-v2sq)*B_1 + xi*(xi2+1)*vdotB*v_1);
  e2 = D*(-(1+xi2)*(v3*b1 - v1*b3) + xi*(1-v2sq)*B_2 + xi*(xi2+1)*vdotB*v_2);
  e3 = D*(-(1+xi2)*(v1*b2 - v2*b1) + xi*(1-v2sq)*B_3 + xi*(xi2+1)*vdotB*v_3);
}

//----------------------------------------------------------------------------------------
//! \fn void chiral::ChiralEMF
//! \brief the non-ideal part of the dynamo EMF, OhmsLawEMF(xi) - OhmsLawEMF(0)
//!
//! Constructive-transport schemes build the ideal EMF from upwinded Riemann
//! fluxes; a non-ideal contribution is then added to the edge field on top of
//! it, as Resistivity::AddEMFConstantResist does with eta*J.  This returns exactly that
//! contribution, so the ideal EMF is never recomputed and never double counted.
//! Arguments and result are as for OhmsLawEMF.
//!
//! At v = 0 this reduces to E_i = xi B_i.
KOKKOS_INLINE_FUNCTION
void ChiralEMF(const Real xi, const Real iW,
               const Real v1, const Real v2, const Real v3,
               const Real b1, const Real b2, const Real b3,
               const Real gxx, const Real gxy, const Real gxz,
               const Real gyy, const Real gyz, const Real gzz,
               const Real isqrtg,
               Real &e1, Real &e2, Real &e3) {
  Real e1_id, e2_id, e3_id;
  OhmsLawEMF(0.0, iW, v1, v2, v3, b1, b2, b3, gxx, gxy, gxz, gyy, gyz, gzz, isqrtg,
             e1_id, e2_id, e3_id);
  OhmsLawEMF(xi, iW, v1, v2, v3, b1, b2, b3, gxx, gxy, gxz, gyy, gyz, gzz, isqrtg,
             e1, e2, e3);
  e1 -= e1_id;
  e2 -= e2_id;
  e3 -= e3_id;
}

//----------------------------------------------------------------------------------------
//! \fn Real chiral::GammaM
//! \brief chirality-flip rate Gamma_m, converted from MeV to code units
//!
//! Gamma_m = alpha^2 m_e^2 / (3 pi mu_e) ln(1/alpha).
//!
//! This is the non-degenerate proton limit, valid for T greater than roughly
//! 13 MeV at nuclear density and Ye = 0.1.  It will underestimate the rate in
//! cooler, more degenerate matter.
//!
//! \param mu_e_MeV     electron chemical potential, MeV
//! \param code_time_s  code time units per second (UnitSystem::time)
KOKKOS_INLINE_FUNCTION
Real GammaM(const Real mu_e_MeV, const Real code_time_s, const Real alpha_em) {
  const Real gamma_m_MeV = SQR(alpha_em) * SQR(kElectronMassMeV)
                           / (3.0 * M_PI * mu_e_MeV)
                           * Kokkos::log(1.0/alpha_em);
  return gamma_m_MeV / (kHbarMeVs * code_time_s);
}

//----------------------------------------------------------------------------------------
//! \fn Real chiral::AnomalySource
//! \brief E.B chiral anomaly source, dn5/dt = (2 alpha / pi) xi b^2 / hbar
//!
//! Heaviside-Lorentz fields, as b is in the code.  xi < 0 for Y5 > 0, so the
//! source drains n5 into the field.
//!
//! \param bsq        fluid-frame b^2 = (B^2 + (W v.B)^2)/W^2
//! \param hbar_code  reduced Planck constant in code units, see HbarCode()
KOKKOS_INLINE_FUNCTION
Real AnomalySource(const Real xi, const Real bsq, const Real alpha_em,
                   const Real hbar_code) {
  return (2.0*alpha_em/M_PI) * xi * bsq / hbar_code;
}

//----------------------------------------------------------------------------------------
//! \fn Real chiral::ImplicitY5Update
//! \brief backward-Euler update of U = D*Y5 under the Gamma_m sink and E.B source
//!
//! Solves A U_new + B cbrt(U_new) = U_old, with A = 1 + dt*alpha*Gamma_m/W (Gamma_m
//! is a rate in proper time and U = sqrt(gamma)*W*n5*mb) and B >= 0 the anomaly
//! coefficient (the E.B source is -B cbrt(U)/dt, since
//! xi is proportional to -cbrt(Y5)).  The source has an infinite derivative at
//! U = 0, so any explicit treatment oscillates there for every B > 0.
//! With u = cbrt(U_new) the left side, A u^3 + B u, is odd and strictly
//! increasing, so the root is unique; min(cbrt(U/A), U/B) bounds it from above
//! and Newton from there decreases monotonically onto it.  B = 0 returns U/A.
KOKKOS_INLINE_FUNCTION
Real ImplicitY5Update(const Real U_old, const Real A, const Real B) {
  if (B <= 0.0 || U_old == 0.0) {
    return U_old/A;
  }
  const Real Ua = Kokkos::fabs(U_old);
  Real u = Kokkos::fmin(Kokkos::cbrt(Ua/A), Ua/B);
  for (int it = 0; it < 64; ++it) {
    const Real du = (A*u*u*u + B*u - Ua)/(3.0*A*u*u + B);
    u -= du;
    if (Kokkos::fabs(du) <= 1.0e-14*u) break;
  }
  return Kokkos::copysign(u*u*u, U_old);
}

//----------------------------------------------------------------------------------------
//! \fn Real chiral::HbarCode
//! \brief reduced Planck constant in code units
//!
//! \param code_energy_per_erg  UnitSystem::energy
//! \param code_time_per_s      UnitSystem::time
KOKKOS_INLINE_FUNCTION
Real HbarCode(const Real code_energy_per_erg, const Real code_time_per_s) {
  return kHbarErgS * code_energy_per_erg * code_time_per_s;
}

}  // namespace chiral

#endif  // MHD_CHIRAL_DYNAMO_HPP_
