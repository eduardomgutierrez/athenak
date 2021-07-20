#ifndef COORDINATES_COORDINATES_HPP_
#define COORDINATES_COORDINATES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file coordinates.hpp
//! \brief implemention of light-weight coordinates class.  Provides data structure that
//! stores array of RegionSizes over (# of MeshBlocks), and inline functions for
//! computing positions.  In GR, also provides inline metric functions (currently only
//! Cartesian Kerr-Schild)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
//! \struct CoordinatesData
//! \brief container for data and inline functions associated with Coordinates class.
//! This includes cell indices, physical locations of MeshBlocks, and functions to compute
//! positions and metric.
//! Storing everything in a container makes it easier to capture coord variables and
//! functions in kernels elsewhere in the code.

struct CoordinatesData
{
  Real bh_mass;                      // needed for GR metric
  Real bh_spin;                      // needed for GR metric
  RegionIndcs mb_indcs;              // indices are the same for all MeshBlocks
  DualArray1D<RegionSize> mb_size;   // array of length (# of MeshBlocks)
  // constructor
  CoordinatesData(int nmb) : mb_size("size",nmb) {}
};

//----------------------------------------------------------------------------------------
//! \class Coordinates
//! \brief data and functions for coordinates

class Coordinates
{
public:
  Coordinates(Mesh *pm, ParameterInput *pin, RegionIndcs indcs, int nmb);
  ~Coordinates() {};

  CoordinatesData coord_data;

  // functions
  void AddCoordTerms(const DvceArray5D<Real> &w0, const EOS_Data &eos, const Real dt,
                     DvceArray5D<Real> &u0);
  void AddCoordTerms(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc,
                     const EOS_Data &eos, const Real dt, DvceArray5D<Real> &u0);

private:
  Mesh* pmy_mesh;
};

#endif // COORDINATES_COORDINATES_HPP_
