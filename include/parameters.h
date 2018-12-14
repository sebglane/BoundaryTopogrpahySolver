/*
 * parameters.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_PARAMETERS_H_
#define INCLUDE_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

namespace TopographyProblem {

using namespace dealii;

struct Parameters
{
    Parameters(const std::string &parameter_filename);
    static void declare_parameters(ParameterHandler &prm);
    void parse_parameters(ParameterHandler &prm);

    void compute_dimensionless_numbers();

    // runtime parameters
    bool    read_dimensional_input;
    bool    homogeneous_bc_at_bottom;

    // mesh parameters
    double amplitude;
    double wave_length;

    // physics parameters
    double buoyancy_frequency;
    double magnetic_diffusivity;

    double reference_velocity;
    double reference_gravity;

    // dimensionless physics parameters
    double Froude;
    double S;

    // linear solver parameters
    double          rel_tol;
    double          abs_tol;
    unsigned int    n_max_iter;

    // discretization parameters
    unsigned int density_degree;
    unsigned int velocity_degree;

    // refinement parameters
    unsigned int n_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
