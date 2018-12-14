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
read_dimensional_input(false),
// geometry parameters
wave_length(1e5),
amplitude(50),
// physics parameters
Froude(1.0  ),
S(1.0),
// linear solver parameters
rel_tol(1e-6),
abs_tol(1e-12),
n_max_iter(100),
// discretization parameters
density_degree(1),
velocity_degree(2),
// refinement parameters
n_refinements(1),
n_initial_refinements(4),
n_boundary_refinements(1),
// entropy viscosity parameters
apply_entropy_viscosity(true),
c_max(0.5),
c_entropy(10.0),
c_velocity(0.5),
default_viscosity(0.1)
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

    prm.enter_subsection("runtime parameters");
    {
        prm.declare_entry("read_dimensional_input",
                          "false",
                          Patterns::Bool(),
                          "program reads dimensional parameter and computes"
                          "dimensionless numbers");
    }
    prm.leave_subsection();

    prm.enter_subsection("dimensional physics parameters");
    {
        prm.declare_entry("buoyancy_frequency",
                "0.729e-4",
                Patterns::Double(0.),
                "buoyancy frequency in 1 / s");

        prm.declare_entry("reference_velocity",
                "5.0e-4",
                Patterns::Double(0.),
                "fluid velocity in m / s");

        prm.declare_entry("reference_gravity",
                "10.0",
                Patterns::Double(0.),
                "gravitional acceleration in m / s^2");
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
            prm.declare_entry("c_max",
                    "0.5",
                    Patterns::Double(0.),
                    "entropy viscosity control parameter");

            prm.declare_entry("c_entropy",
                    "1.0",
                    Patterns::Double(0.),
                    "entropy viscosity control parameter");

            prm.declare_entry("c_velocity",
                    "0.5",
                    Patterns::Double(0.),
                    "viscosity control parameter for velocitys");

            prm.declare_entry("apply_entropy_viscosity",
                    "false",
                    Patterns::Bool(),
                    "default viscosity applied in initial step");

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

    prm.enter_subsection("dimensionless physics parameters");
    {
        prm.declare_entry("Froude",
                "1.0",
                Patterns::Double(0.),
                "Froude number");

        prm.declare_entry("stratification_number",
                "1.0e-3",
                Patterns::Double(),
                "dimensionless number describing strength of stratification: N^2 l / g ");

    }
    prm.leave_subsection();

//    prm.enter_subsection("linear solver settings");
//    {
//        prm.declare_entry("tol_rel",
//                "1e-6",
//                Patterns::Double(),
//                "relative tolerance for the linear solver.");
//
//        prm.declare_entry("tol_abs",
//                "1e-12",
//                Patterns::Double(),
//                "absolute tolerance for the linear solver.");
//
//        prm.declare_entry("n_max_iter",
//                "100",
//                Patterns::Integer(0),
//                "maximum number of iterations for the linear solver.");
//    }
//    prm.leave_subsection();
}

void Parameters::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("geometry parameters");
    {
        wave_length = prm.get_double("wave_length");
        amplitude = prm.get_double("amplitude");
    }
    prm.leave_subsection();

    prm.enter_subsection("runtime parameters");
    {
        read_dimensional_input = prm.get_bool("read_dimensional_input");
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
            apply_entropy_viscosity = prm.get_bool("apply_entropy_viscosity");
            c_max = prm.get_double("c_max");
            c_entropy = prm.get_double("c_entropy");
            c_velocity = prm.get_double("c_velocity");
            default_viscosity =  prm.get_double("default_viscosity");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    if (read_dimensional_input)
    {

        prm.enter_subsection("dimensional physics parameters");
        {
            buoyancy_frequency = prm.get_double("buoyancy_frequency");

            reference_velocity = prm.get_double("reference_velocity");
            reference_gravity = prm.get_double("reference_gravity");

            compute_dimensionless_numbers();
        }
        prm.leave_subsection();
    }
    else
    {
        prm.enter_subsection("dimensionless physics parameters");
        {
            Froude = prm.get_double("Froude");
            S = prm.get_double("stratification_number");

        }
        prm.leave_subsection();
    }

//    prm.enter_subsection("linear solver settings");
//    {
//        rel_tol = prm.get_double("tol_rel");
//        abs_tol = prm.get_double("tol_abs");
//
//        n_max_iter = prm.get_integer("n_max_iter");
//    }
//    prm.leave_subsection();
}

void Parameters::compute_dimensionless_numbers()
{
    Froude = reference_velocity / std::sqrt(wave_length * reference_gravity);
    S = buoyancy_frequency * buoyancy_frequency * wave_length / reference_gravity;
}


}  // namespace BuoyantFluid
