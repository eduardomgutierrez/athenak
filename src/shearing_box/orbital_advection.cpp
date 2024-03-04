//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables in the orbital advection step used with the shearing box. Data is shifted
//! by the appropriate offset during the recv/unpack step, so these functions both
//! communicate the data and perform the shift. Based on BoundaryValues send/recv funcs.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendCC_Orb()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors for
//! the orbital advection step.  Only ghost zones on the x2-faces (Y-faces) are passed.
//! Communication of coarse arrays is not needed since variable resolution in x2 is not
//! allowed in this shearing box implementation.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus ShearingBox::PackAndSendCC_Orb(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf_orb;
  auto &rbuf = recvbuf_orb;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*2*nvar;  // only consider 2 neighbors (x2-faces)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-pack", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(2*nvar);
    const int n = (tmember.league_rank() - m*(2*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(2*nvar) - n*nvar);

    // indices of x2-face buffers in nghbr view
    int nnghbr;
    if (n==0) {nnghbr=8;} else {nnghbr=12;}

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      // neighbor must always be at same level, so use same indices to pack buffer
      // Note j-range of indices extended by shear
      int il = is;
      int iu = ie;
      int jl, ju;
      if (n==0) {
        jl = js;
        ju = js + (ng + maxjshift - 1);;
      } else {
        jl = je - (ng + maxjshift - 1);
        ju = je;
      }
      int kl = ks;
      int ku = ke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nji = nj*ni;
      int nkji = nk*nj*ni;

      // index of recv'ing (destination) MB and buffer [0,1]: MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,nnghbr).gid - mbgid.d_view(0);
      int dn = (n+1) % 2;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;

        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          rbuf[dn].vars(dm,v,k-kl,j-jl,i-il) = a(m,v,k,j,i);

        // else copy into send buffer for MPI communication below
        } else {
          sbuf[n].vars(m,v,k-kl,j-jl,i-il) = a(m,v,k,j,i);
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=8; n<=12; n+=4) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          auto send_ptr = Kokkos::subview(sbuf[n].vars, m, Kokkos::ALL, Kokkos::ALL,
                                          Kokkos::ALL, Kokkos::ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb, &(sbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
// \!fn void RecvAndUnpackCC_Orb()
// \brief Receive and unpack boundary buffers for CC variables with orbital advection.
//! Cell-centered variables in input array u0 are remapped during unpack by applying
//! both an integer shift and a fractional offset.

TaskStatus ShearingBox::RecvAndUnpackCC_Orb(DvceArray5D<Real> &u0,
                                            ReconstructionMethod rcon){
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf_orb;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack and apply shift

  int nvar = u0.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int ncells2 = indcs.nx2 + 2*(indcs.ng);

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qom = qshear*omega0;
  Real ly = (mesh_size.x2max - mesh_size.x2min);

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells2) * 3;
  par_for_outer("oa-unpk",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> u0_(member.team_scratch(scr_lvl), ncells2); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), ncells2); // "U_star" at faces
    ScrArray1D<Real> q1_(member.team_scratch(scr_lvl), ncells2); // scratch array

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -qom*x1v*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array with integer shift such that j -> jj - joffset
    par_for_inner(member, 0, (ncells2-1), [&](const int jj) {
      if ((jj-joffset) < js) {
        // Load from L boundary buffer with offset
        u0_(jj) = rbuf[0].vars(m,n,k-ks,((jj-joffset)+maxjshift),i-is);
      } else if ((jj-joffset) < (je+1)) {
        // Load from conserved variables themselves with offset
        u0_(jj) = u0(m,n,k,(jj-joffset),i);
      } else {
        // Load from R boundary buffer with offset
        u0_(jj) = rbuf[1].vars(m,n,k-ks,((jj-joffset)-(je+1)),i-is);
      }
    });
    member.team_barrier();

    // Compute "fluxes" of shifted array (u0_) for remap by remainder
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DonorCellOrbAdvFlx(member, js, je+1, epsi, u0_, q1_, flx);
        break;
      case ReconstructionMethod::plm:
        PcwsLinearOrbAdvFlx(member, js, je+1, epsi, u0_, q1_, flx);
        break;
//      case ReconstructionMethod::ppm4:
//      case ReconstructionMethod::ppmx:
//          PiecewiseParabolicOrbAdvFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
//        break;
      default:
        break;
    }
    member.team_barrier();

    // Update CC variables with both integer shift (from u0_) and a conservative remap
    // for the remaining fraction of a cell using upwind "fluxes"
    par_for_inner(member, js, je, [&](const int j) {
      u0(m,n,k,j,i) = u0_(j) - (flx(j+1) - flx(j));
    });
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendFC_Orb()
//! \brief Pack face-centered fields into boundary buffers and send to neighbors for
//! the orbital advection step. Only ghost zones on the x2-faces (Y-faces) are passed.
//! Note only B3 and B1 need be passed.

TaskStatus ShearingBox::PackAndSendFC_Orb(DvceFaceFld4D<Real> &b) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &mblev = pmy_pack->pmb->mb_lev;
  auto &sbuf = sendbuf_orb;
  auto &rbuf = recvbuf_orb;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*2;  // only consider 2 neighbors (x2-faces)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-packB", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = tmember.league_rank()/2;
    const int n = tmember.league_rank()%2;

    // indices of x2-face buffers in nghbr view
    int nnghbr;
    if (n==0) {nnghbr=8;} else {nnghbr=12;}

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,nnghbr).gid >= 0) {
      // neighbor must always be at same level, so use same indices to pack buffer
      // Note j-range of indices extended by shear
      int il = is;
      int iu = ie+1;
      int jl, ju;
      if (n==0) {
        jl = js;
        ju = js + (ng + maxjshift - 1);;
      } else {
        jl = je - (ng + maxjshift - 1);
        ju = je;
      }
      int kl = ks;
      int ku = ke+1;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nji = nj*ni;
      int nkji = nk*nj*ni;

      // index of recv'ing (destination) MB and buffer [0,1]: MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,nnghbr).gid - mbgid.d_view(0);
      int dn = (n+1) % 2;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;

        // copy B1/B3 directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          rbuf[dn].flds(dm,0,k-kl,j-jl,i-il) = b.x3f(m,k,j,i);
          rbuf[dn].flds(dm,1,k-kl,j-jl,i-il) = b.x1f(m,k,j,i);

        // else copy B1/B3 into send buffer for MPI communication below
        } else {
          sbuf[n].flds(m,0,k-kl,j-jl,i-il) = b.x3f(m,k,j,i);
          sbuf[n].flds(m,1,k-kl,j-jl,i-il) = b.x1f(m,k,j,i);
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=8; n<=12; n+=4) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          auto send_ptr = Kokkos::subview(sbuf[n].flds, m, Kokkos::ALL, Kokkos::ALL,
                                          Kokkos::ALL, Kokkos::ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb, &(sbuf[n].flds_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void RecvAndUnpackFC_Orb()
//! \brief Receive and unpack boundary buffers for FC fields with orbital advection.
//! Since CT is required to update fields, the algorithm used here works somewhat
//! differently than that used for CC variables in RecvAndUnpackCC_Orb(). Here an
//! effective electric field is computed including both the integer and fractional cell
//! shifts.  These fields are then used to update B using CT. The fields themselves are
//! not directly remapped like the CC variables.

TaskStatus ShearingBox::RecvAndUnpackFC_Orb(DvceFaceFld4D<Real> &b0,
                                            ReconstructionMethod rcon){
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf_orb;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].flds_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack and compute effective EMF

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int ncells2 = indcs.nx2 + 2*(indcs.ng);

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qom = qshear*omega0;
  Real ly = (mesh_size.x2max - mesh_size.x2min);

  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(ncells2) * 3;
  auto emfx = pmy_pack->pmhd->efld.x1e;
  auto emfz = pmy_pack->pmhd->efld.x3e;
  par_for_outer("oa-unB",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,1,ks,ke+1,is,ie+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int v, const int k, const int i) {
    ScrArray1D<Real> u0_(member.team_scratch(scr_lvl), ncells2); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), ncells2); // "U_star" at faces
    ScrArray1D<Real> q1_(member.team_scratch(scr_lvl), ncells2); // scratch array

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;

    Real x1;
    if (v==0) {
      // B3 located at x1-cell centers
      Real x1 = CellCenterX(i-is, nx1, x1min, x1max);
    } else if (v==1) {
      // B1 located at x1-cell faces
      Real x1 = LeftEdgeX(i-is, nx1, x1min, x1max);
    }
    Real yshear = -qom*x1*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array.  Index with shift:  jj = j + jshift
    par_for_inner(member, 0, (ncells2-1), [&](const int jj) {
      if ((jj-joffset) < js) {
        // Load scratch arrays from L boundary buffer with offset
        u0_(jj) = rbuf[0].vars(m,v,k-ks,((jj-joffset)+maxjshift),i-is);
      } else if ((jj-joffset) < (je+1)) {
        // Load from array itself with offset
        if (v==0) {
          u0_(jj) = b0.x3f(m,k,(jj-joffset),i);
        } else if (v==1) {
          u0_(jj) = b0.x1f(m,k,(jj-joffset),i);
        }
      } else {
        // Load scratch arrays from R boundary buffer with offset
        u0_(jj) = rbuf[1].vars(m,v,k-ks,((jj-joffset)-(je+1)),i-is);
      }
    });
    member.team_barrier();

    // Compute x2-fluxes from fractional offset
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DonorCellOrbAdvFlx(member, js, je+1, epsi, u0_, q1_, flx);
        break;
      case ReconstructionMethod::plm:
        PcwsLinearOrbAdvFlx(member, js, je+1, epsi, u0_, q1_, flx);
        break;
//      case ReconstructionMethod::ppm4:
//      case ReconstructionMethod::ppmx:
//          PiecewiseParabolicOrbAdvFlx(member,eos_,extrema,true,m,k,j,il,iu, w0_, wl_jp1, wr);
//        break;
      default:
        break;
    }
    member.team_barrier();

    // Compute emfx = -VyBz, which is at cell-center in x1-direction
    if (v==0) {
      par_for_inner(member, js, je, [&](const int j) {
        emfx(m,k,j,i) = -flx(j);
      });
      // Sum integer offsets into effective EMFs
      for (int j=1; j<=joffset; j++) {
        emfx(m,k,j,i) -= u0_(j);
      }
      for (int j=(joffset+1); j<=0; j++) {
        emfx(m,k,j,i) += u0_(j);
      }

    // Compute emfz =  VyBx, which is at cell-face in x1-direction
    } else if (v==1) {
      par_for_inner(member, js, je, [&](const int j) {
        emfz(m,k,j,i) = flx(j);
      });
      // Sum integer offsets into effective EMFs
      for (int j=1; j<=joffset; j++) {
        emfz(m,k,j,i) += u0_(j);
      }
      for (int j=(joffset+1); j<=0; j++) {
        emfz(m,k,j,i) -= u0_(j);
      }
    }
  });

  // Update face-centered fields using CT
  //---- update B1 (only for 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b1", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) -= (emfz(m,k,j+1,i) - emfz(m,k,j,i));
    });
  }

  //---- update B2 (curl terms in 1D and 3D problems)
  par_for("oaCT-b2", DevExeSpace(), 0, nmb-1, ks, ke, js, je+1, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real dydx = mbsize.d_view(m).dx2/mbsize.d_view(m).dx1;
    b0.x2f(m,k,j,i) += dydx*(emfz(m,k,j,i+1) - emfz(m,k,j,i));
    if (pmy_pack->pmesh->three_d) {
      Real dydz = mbsize.d_view(m).dx2/mbsize.d_view(m).dx3;
      b0.x2f(m,k,j,i) -= dydz*(emfx(m,k+1,j,i) - emfx(m,k,j,i));
    }
  });

  //---- update B3 (curl terms in 1D and 2D/3D problems)
  if (pmy_pack->pmesh->multi_d) {
    par_for("oaCT-b3", DevExeSpace(), 0, nmb-1, ks, ke+1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x3f(m,k,j,i) += (emfx(m,k,j+1,i) - emfx(m,k,j,i));
    });
  }

  return TaskStatus::complete;
}
