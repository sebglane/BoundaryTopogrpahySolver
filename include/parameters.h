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


    // mesh parameters
    double amplitude;
    double wave_length;

    // discretization parameters
    unsigned int velocity_degree;

    // entropy viscosity parameters
    double  c_velocity;
    double  default_viscosity;

    // newton iteration
    double          tolerance;
    unsigned int    max_iter;

    // refinement parameters
    unsigned int n_refinements;
    unsigned int n_initial_refinements;
    unsigned int n_boundary_refinements;
};


}  // namespace BuoyantFluid

#endif /* INCLUDE_PARAMETERS_H_ */
