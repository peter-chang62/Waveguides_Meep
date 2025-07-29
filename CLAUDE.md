# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment Setup

This codebase uses Python with the `idp` virtual environment as specified in `pyrightconfig.json`. Key dependencies include:
- MEEP (electromagnetic simulation)
- MPB (MIT Photonic Bands)
- NumPy, SciPy, matplotlib
- PyNLO (for nonlinear optics calculations)

## Project Structure

This is a photonic waveguide simulation package built on MEEP/MPB for calculating electromagnetic wave propagation in thin-film waveguides:

### Core Package (`TFW_meep/`)
- `geometry.py`: Geometric transformations for waveguide structures (e.g., converting rectangular blocks to trapezoidal etch profiles)
- `materials.py`: Optical material definitions using Lorentzian susceptibilities (currently contains Al2O3/sapphire)
- `waveguide_dispersion.py`: Main simulation classes:
  - `RidgeWaveguide`: Base class for rectangular waveguides on infinite substrates
  - `ThinFilmWaveguide`: Extends RidgeWaveguide for etched thin-film structures

### Examples and Usage (`examples/`)
- Example scripts showing how to calculate dispersion for specific material systems
- Integration with PyNLO for nonlinear optics calculations
- Custom epsilon functions for materials not built into MEEP (e.g., lithium niobate using Gayer/Jundt equations)

## Key Simulation Concepts

**Dispersion Calculation Methods:**
- `calc_dispersion()`: Calculate k-vector given frequency (ω → k)
- `calc_w_from_k()`: Calculate frequency given k-vector (k → ω)
- `find_k()`: Core MPB interface for finding modes at specific frequencies

**Etch Angle Simulations:**
- Use `@etch_angle_sim_wrapper` decorator to temporarily convert rectangular waveguides to trapezoidal cross-sections
- Default etch angle is 80 degrees

**Material Handling:**
- MPB doesn't support dispersive materials directly
- Workaround: Set fixed epsilon for each frequency point during iteration
- Custom epsilon functions (`eps_func_wvgd`, `eps_func_sbstrt`) can override MEEP's built-in material models

## Development Notes

- Pyright configuration disables several type checking warnings due to MEEP's dynamic nature
- The codebase includes extensive simulation output storage in `sim_output/` and visualization in `gif/`
- Mode visualization includes effective mode area calculations using Simpson integration
- Field storage (E, H) and group velocity calculations are built into the dispersion routines

## Working with Geometric Objects

When modifying waveguide geometry:
1. Save original `blk_wvgd` before substitution
2. Apply geometric transformation
3. Run simulation
4. Restore original geometry to maintain property setters/getters functionality