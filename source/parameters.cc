/*
 * parameters.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include "parameters.h"

namespace TopographyProblem {

Parameters::Parameters(const std::string &parameter_filename)
:
// geometry parameters
wave_length(1e5),
amplitude(50),
// discretization parameters
density_degree(1),
velocity_degree(2),
// refinement parameters
n_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
// entropy viscosity parameters
c_velocity(0.5),
default_viscosity(0.1),
// newton iteration control
tolerance(1e-12),
max_iter(15)
{
    ParameterHandler prm;
    declare_parameters(prm);

    std::ifstream parameter_file(parameter_filename.c_str());

    if (!parameter_file)
    {
        parameter_file.close();

        std::ostringstream message;
        message << "Input parameter file <"
                << parameter_filename << "> not found. Creating a"
                << std::endl
                << "template file of the same name."
                << std::endl;

        std::ofstream parameter_out(parameter_filename.c_str());
        prm.print_parameters(parameter_out,
                ParameterHandler::OutputStyle::Text);

        AssertThrow(false, ExcMessage(message.str().c_str()));
    }

    prm.parse_input(parameter_file);

    parse_parameters(prm);
}


void Parameters::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("geometry parameters");
    {
        prm.declare_entry("wave_length",
                          "1e5",
                          Patterns::Double(0.),
                          "wave length of the topography");

        prm.declare_entry("amplitude",
                          "50",
                          Patterns::Double(0.),
                          "amplitude of the topography");

    }
    prm.leave_subsection();

    prm.enter_subsection("discretization parameters");
    {
        prm.declare_entry("velocity_degree",
                "2",
                Patterns::Integer(1,2),
                "Polynomial degree of the velocity discretization. The polynomial "
                "degree of the pressure is automatically set to one less than the velocity.");

        prm.declare_entry("density_degree",
                "1",
                Patterns::Integer(1,2),
                "Polynomial degree of the density discretization.");

        prm.enter_subsection("entropy viscosity parameters");
        {
            prm.declare_entry("c_velocity",
                    "0.5",
                    Patterns::Double(0.),
                    "viscosity control parameter for velocitys");


            prm.declare_entry("default_viscosity",
                    "0.1",
                    Patterns::Double(0.),
                    "default viscosity applied in initial step");
        }
        prm.leave_subsection();

        prm.enter_subsection("refinement parameters");
        {
            prm.declare_entry("n_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of refinements.");

            prm.declare_entry("n_initial_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of initial refinements");

            prm.declare_entry("n_boundary_refinements",
                    "1",
                    Patterns::Integer(),
                    "number of initial boundary refinements");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Newton iteration control");
    {
        prm.declare_entry("tolerance",
                "1e-6",
                Patterns::Double(),
                "tolerance for termination of Newton iteration");

        prm.declare_entry("max_iter",
                "15",
                Patterns::Integer(0),
                "maximum number of iterations for the Newton solver");
    }
    prm.leave_subsection();
}

void Parameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("geometry parameters");
    {
        wave_length = prm.get_double("wave_length");
        amplitude = prm.get_double("amplitude");
    }
    prm.leave_subsection();

    prm.enter_subsection("discretization parameters");
    {
        velocity_degree = prm.get_integer("velocity_degree");
        density_degree = prm.get_integer("density_degree");

        prm.enter_subsection("refinement parameters");
        {
            n_refinements = prm.get_integer("n_refinements");
            n_initial_refinements = prm.get_integer("n_initial_refinements");
            n_boundary_refinements = prm.get_integer("n_boundary_refinements");
        }
        prm.leave_subsection();

        prm.enter_subsection("entropy viscosity parameters");
        {
            c_velocity = prm.get_double("c_velocity");
            default_viscosity =  prm.get_double("default_viscosity");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    prm.enter_subsection("Newton iteration control");
    {
        tolerance = prm.get_double("tolerance");
        max_iter = prm.get_integer("max_iter");
    }
    prm.leave_subsection();
}
}  // namespace BuoyantFluid
