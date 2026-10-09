#!/usr/bin/env bash
# Validation suite for the multifrequency neutrino Boltzmann solver, against grey M1 and
# analytic solutions.  Every run is kept under ~5 minutes on one CPU core, cfl >= 0.01.
#
#   eq_flat     single-zone relaxation to trapped beta-equilibrium, four matter states
#   eq_nfreq    the equilibrium at nfreq 12/48 (24 is in eq_flat)
#   eq_boost    the same with the fluid boosted to v = 0.3
#   eq_curved   the same at rest at r = 20/10/6/4 M of a Schwarzschild black hole
#   freestream  hot wall radiating into vacuum (flat; against the analytic 1 - x/ct)
#   diffusion   spectrum diffusing through a moving scattering medium
#   redshift    free-streaming gravitational redshift (frequency fluxes, curved; neutrino path)
#   lepton      lepton-number budget of the Ye source (flat, at rest and boosted)
#
# Usage:  scripts/nu_validation/run_suite.sh TABLE [OUTDIR]
# Needs the builds (from the repository root):
#   build     -D Athena_ENABLE_NURATES=ON                                  (M1 pgens)
#   build-sz  -D Athena_ENABLE_NURATES=ON -D PROBLEM=rad_neutrino_singlezone
#   build-fr  -D Athena_ENABLE_NURATES=ON -D PROBLEM=rad_freq_redshift
#   build-nonu (no nurates; the M1 diffusion pgen needs a single species)
# Plots and the summary table:  vis/python/nu_validation_plots.py OUTDIR
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TABLE=${1:?usage: run_suite.sh TABLE [OUTDIR]}
OUT=${2:-$ROOT/runs/nu_validation}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)   # absolute: run() cds into it
MF=$ROOT/build-sz/src/athena
BI=$ROOT/build/src/athena
FR=$ROOT/build-fr/src/athena
NONU=$ROOT/build-nonu/src/athena
TESTS=$ROOT/inputs/tests
mkdir -p $OUT/inputs

# ---- local input copies (command-line overrides cannot add keys) ----------------------
sed -e 's/^use_dm_eff = false/use_dm_eff = false\nuse_decay = false/' \
    -e 's/^Y_e = 0.1 .*/Y_e = 0.1\nadm_from_coord = true/' \
    $TESTS/rad_m1_singlezone.athinput > $OUT/inputs/m1_curved.athinput
sed -e 's/^excise      = false/excise      = false\nminkowski   = true/' \
    $OUT/inputs/m1_curved.athinput > $OUT/inputs/m1_flat.athinput
sed -e 's/^minkowski   = true/a           = 0.0\nexcise      = false/' \
    $TESTS/rad_neutrino_singlezone.athinput > $OUT/inputs/mf_curved.athinput
cp $TESTS/rad_diffusion_spectral_nurates.athinput $OUT/inputs/mf_diffusion.athinput
printf '\n<output1>\nfile_type = bin\nvariable  = radnu_fluid\ndt        = 0.25\n' \
    >> $OUT/inputs/mf_diffusion.athinput
printf '\n<output2>\nfile_type = tab\nvariable  = rad_coord_fluid\ndt        = 0.25\n' \
    >> $OUT/inputs/mf_diffusion.athinput

# Lepton-budget inputs: the pair and bremsstrahlung channels do not conserve lepton
# number as implemented, so they are switched off here (they stay on elsewhere).
sed -e 's/^use_inelastic_scatt = false/use_inelastic_scatt = false\nuse_pair = false\nuse_brem = false/' \
    $TESTS/rad_neutrino_singlezone.athinput > $OUT/inputs/mf_lepton.athinput
sed -e 's/^use_inelastic_scatt = false/use_inelastic_scatt = false\nuse_pair = false\nuse_brem = false\nuse_decay = false/' \
    $OUT/inputs/m1_flat.athinput > $OUT/inputs/m1_lepton.athinput

run() {  # run DIR BINARY INPUT [overrides...]
  local dir=$1 bin=$2 in=$3; shift 3
  mkdir -p $OUT/$dir && cd $OUT/$dir && rm -rf tab bin *.hst *.txt
  $bin -i $in "$@" > log.txt 2>&1
  echo "$dir exit=$?"
}

# ---- 1. single-zone equilibration, flat --------------------------------------------------
# name:rho[code]:T0[MeV]:Ye0:cfl:tlim:box half-width:nu_min:nu_max.  dt = cfl*2*hw/5; the
# zone is uniform and periodic, so the box only sets the time step.
STATES="A:1.619e-5:50.0:0.1:0.01:3:2.5:1.0:3000.0
        D13:1.619e-5:10.0:0.3:0.5:3000:2.5:0.5:300.0
        D12:1.619e-6:10.0:0.3:0.5:3000:2.5:0.5:300.0
        N11:1.619e-7:5.0:0.3:0.5:100000:100:0.5:300.0"
for s in $STATES; do
  IFS=: read -r name rho T Y cfl tlim hw numin numax <<< "$s"
  nt=$(python3 -c "print(max(1, int($tlim/($cfl*2*$hw/5)/300)))")
  common="mhd/table=$TABLE problem/rho=$rho problem/temp=$T problem/Y_e=$Y time/cfl_number=$cfl
          time/tlim=$tlim mesh/x1min=-$hw mesh/x1max=$hw"
  run eq_flat/$name/mf $MF $TESTS/rad_neutrino_singlezone.athinput $common \
      radiation/nfreq=24 radiation/nu_min=$numin radiation/nu_max=$numax \
      output2/dcycle=$nt output3/dcycle=$nt output4/dcycle=$((nt*30)) &
  run eq_flat/$name/m1 $BI $OUT/inputs/m1_flat.athinput $common time/nlim=-1 \
      output1/dcycle=$nt output2/dcycle=$nt output3/dcycle=$nt &
done

# equilibrium vs frequency resolution (the bin quadrature of the neutrino lepton number)
for nf in 12 48; do
  run eq_nfreq/A/nf$nf $MF $TESTS/rad_neutrino_singlezone.athinput mhd/table=$TABLE \
      problem/rho=1.619e-5 problem/temp=50.0 problem/Y_e=0.1 time/cfl_number=0.01 time/tlim=3 \
      radiation/nfreq=$nf output2/dcycle=10 output3/dcycle=10 output4/dcycle=100000 &
  run eq_nfreq/D13/nf$nf $MF $TESTS/rad_neutrino_singlezone.athinput mhd/table=$TABLE \
      problem/rho=1.619e-5 problem/temp=10.0 problem/Y_e=0.3 time/cfl_number=0.5 time/tlim=3000 \
      radiation/nfreq=$nf radiation/nu_min=0.5 radiation/nu_max=300.0 \
      output2/dcycle=50 output3/dcycle=50 output4/dcycle=100000 &
done

# ---- 2. boosted single zone (state A, v = 0.3) ------------------------------------------
BA="mhd/table=$TABLE problem/rho=1.619e-5 problem/temp=50.0 problem/Y_e=0.1 problem/vx=0.3
    time/cfl_number=0.01 time/tlim=3.0"
run eq_boost/mf $MF $TESTS/rad_neutrino_singlezone.athinput $BA radiation/nfreq=24 \
    output2/dcycle=3 output3/dcycle=3 output4/dcycle=60 &
run eq_boost/m1 $BI $OUT/inputs/m1_flat.athinput $BA time/nlim=-1 \
    output1/dcycle=3 output2/dcycle=3 output3/dcycle=3 &

# ---- 3. curved single zone (state A at rest at r, box 0.05 M; flat reference) ---------------------------
CA="mhd/table=$TABLE problem/rho=1.619e-5 problem/temp=50.0 problem/Y_e=0.1
    time/cfl_number=0.5 time/tlim=1.0 mesh/nx1=5"
for r in 20 10 6 4; do
  lo=$(python3 -c "print($r-0.025)"); hi=$(python3 -c "print($r+0.025)")
  run eq_curved/r$r/mf $MF $OUT/inputs/mf_curved.athinput $CA radiation/nfreq=24 \
      mesh/x1min=$lo mesh/x1max=$hi meshblock/nx1=5 output2/dcycle=2 output3/dcycle=2 \
      output4/dcycle=50 &
  run eq_curved/r$r/m1 $BI $OUT/inputs/m1_curved.athinput $CA time/nlim=-1 \
      radiation_m1/gr_sources=true mesh/x1min=$lo mesh/x1max=$hi \
      output1/dcycle=2 output2/dcycle=2 output3/dcycle=2 &
done
run eq_curved/flat/mf $MF $TESTS/rad_neutrino_singlezone.athinput $CA radiation/nfreq=24 \
    mesh/x1min=5.975 mesh/x1max=6.025 meshblock/nx1=5 output2/dcycle=2 output3/dcycle=2 \
    output4/dcycle=50 &
run eq_curved/flat/m1 $BI $OUT/inputs/m1_flat.athinput $CA time/nlim=-1 \
    radiation_m1/gr_sources=true mesh/x1min=5.975 mesh/x1max=6.025 \
    output1/dcycle=2 output2/dcycle=2 output3/dcycle=2 &
wait

# ---- 4. free streaming from a hot wall (flat) --------------------------------------------
run freestream/mf $FR $TESTS/rad_freq_redshift.athinput coord/minkowski=true \
    mesh/x1min=0.0 mesh/x1max=1.0 mesh/nx1=128 meshblock/nx1=128 time/tlim=0.75 \
    output1/dt=0.25 radiation/nlevel=4 radiation/nfreq=4 radiation/radiation_type=neutrino \
    radiation/use_nurates=true radiation/freq_fluxes=false &

# ---- 5. diffusion through a moving scattering medium (v = 0.3) -------------------------
run diffusion/mf $BI $OUT/inputs/mf_diffusion.athinput radiation/nfreq=24 &
run diffusion/m1 $NONU $TESTS/rad_m1_diffusiontest.athinput problem/fluid_velocity=0.3 \
    time/tlim=1.0 mesh/nx1=128 &

# ---- 6. free-streaming gravitational redshift (neutrino path) --------------------------
# nu_min sits a decade below the injected spectrum: first-order frequency advection
# broadens it as it shifts, and what reaches nu_min leaves the grid, which would bias the
# measured mean energy of what remains.
run redshift/mf $FR $TESTS/rad_freq_redshift.athinput mesh/x1max=23.0 mesh/nx1=80 \
    meshblock/nx1=80 time/tlim=40.0 radiation/freq_fluxes=true radiation/nu_min=0.03 \
    radiation/nu_max=30.0 radiation/nlevel=3 radiation/nfreq=36 \
    radiation/radiation_type=neutrino radiation/use_nurates=true &

# ---- 7. lepton-number budget of the Ye source ------------------------------------------
# Uniform periodic zone, so there is no transport: the electron number the matter gains
# must equal the neutrino number the field loses, Delta(rho W Ye)/m_b = -Delta(N_nue,E -
# N_anue,E), both read from the outputs.  Run at rest and boosted.
LB="mhd/table=$TABLE problem/rho=1.619e-5 problem/temp=50.0 problem/Y_e=0.1
    time/cfl_number=0.01 time/tlim=0.05"
for v in 0.0 0.3; do
  run lepton/v$v/mf $MF $OUT/inputs/mf_lepton.athinput $LB problem/vx=$v \
      radiation/nfreq=24 output2/dcycle=1 output3/dcycle=1 \
      output4/variable=radnu_coord output4/dcycle=1 &
  run lepton/v$v/m1 $BI $OUT/inputs/m1_lepton.athinput $LB problem/vx=$v time/nlim=-1 \
      output1/dcycle=1 output2/dcycle=1 output3/variable=rad_m1_N output3/dcycle=1 &
done
wait
echo "suite done: $OUT"
