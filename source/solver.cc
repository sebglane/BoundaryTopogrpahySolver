/*
 * solver.cc
 *
 *  Created on: Nov 8, 2018
 *      Author: sg
 */

#include <deal.II/base/parameter_handler.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include "solver.h"
#include "postprocessor.h"

namespace TopographyProblem {

template<int dim>
TopographySolver<dim>::TopographySolver(Parameters &parameters_)
:
parameters(parameters_),
background_velocity_value(Point<dim>::unit_vector(0)),
background_velocity_gradient(),
// triangulation
triangulation(),
// finite element part
fe_system(FESystem<dim>(FE_Q<dim>(parameters.velocity_degree), dim), 1,
          FE_Q<dim>(parameters.velocity_degree - 1), 1),
dof_handler(triangulation),
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{
    std::cout << "Topography solver by S. Glane\n"
              << "This program solves inviscid flow over topography.\n"
              << "The governing equations are\n\n"
              << "\t-- Incompressibility constraint:\n\t\t div(v) = 0,\n\n"
              << "\t-- Navier-Stokes equation:\n\t\t V . grad(v) + v . grad(V)\n"
              << "\t\t\t\t= - grad(p),\n\n";

   std::cout << std::endl << "You have chosen the following parameter set:";

   std::stringstream ss;
   ss << "+----------+----------+\n"
      << "|    k     |    h     |\n"
      << "+----------+----------+\n"
      << "| "
      << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.wave_length
      << " | "
      << std::setw(8) << std::setprecision(1) << std::scientific << std::right << parameters.amplitude
      << " |\n"
      << "+----------+----------+\n";

   std::cout << std::endl << ss.str() << std::endl;
   std::cout << std::endl << std::fixed << std::flush;
}

template<int dim>
void TopographySolver<dim>::output_results(const unsigned int level) const
{
    std::cout << "   Output results..." << std::endl;

    PostProcessor<dim>  postprocessor;

    // prepare data out object
    DataOut<dim, DoFHandler<dim>>    data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution, postprocessor);

    // compute cell viscosity
    Vector<double>  cell_viscosity_velocity(triangulation.n_active_cells());
    {
        QMidpoint<dim>      quadrature;

        FEValues<dim>       fe_values(fe_system,
                                      quadrature,
                                      update_values);

        const unsigned int n_q_points    = quadrature.size();

        // momentum part
        std::vector<Tensor<1,dim>>  present_velocity_values(n_q_points);
        std::vector<double>         present_velocity_divergences(n_q_points);

        const FEValuesExtractors::Vector    velocity(0);

        for (auto cell: dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            fe_values[velocity].get_function_values(present_solution,
                                                    present_velocity_values);
            // viscosity momentum equation
            const double nu_velocity = compute_velocity_viscosity(present_velocity_values,
                                                                  cell->diameter());
            cell_viscosity_velocity(cell->index()) = nu_velocity;
        }
    }
    data_out.add_data_vector(cell_viscosity_velocity,
                             "cell_viscosity_velocity");

    data_out.build_patches(parameters.velocity_degree - 1);

    // write output to disk
    const std::string filename = ("solution-" +
                                  Utilities::int_to_string(level, 2) +
                                  ".vtk");
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
}

template<int dim>
void TopographySolver<dim>::refine_mesh()
{
    TimerOutput::Scope timer_section(computing_timer, "refine mesh");

    std::cout << "   Mesh refinement..." << std::endl;

    // error estimation based on temperature
    Vector<float>   estimated_error_per_cell(triangulation.n_active_cells());
    const FEValuesExtractors::Vector    velocity(1);

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim-1>(parameters.velocity_degree + 1),
                                       typename FunctionMap<dim>::type(),
                                       present_solution,
                                       estimated_error_per_cell,
                                       fe_system.component_mask(velocity));
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.8, 0.0);

    // preparing temperature solution transfer
    std::vector<BlockVector<double>> x_solution(1);
    x_solution[0] = present_solution;
    SolutionTransfer<dim,BlockVector<double>> solution_transfer(dof_handler);

    // preparing triangulation refinement
    triangulation.prepare_coarsening_and_refinement();
    solution_transfer.prepare_for_coarsening_and_refinement(x_solution);

    // refine triangulation
    triangulation.execute_coarsening_and_refinement();

    // setup dofs and constraints on refined mesh
    setup_dofs();

    // transfer of solution
    {
        std::vector<BlockVector<double>> tmp_solution(1);
        tmp_solution[0].reinit(present_solution);
        solution_transfer.interpolate(x_solution, tmp_solution);

        present_solution = tmp_solution[0];

        nonzero_constraints.distribute(present_solution);
    }
}

template<int dim>
void TopographySolver<dim>::run()
{
    bool initial_step = true;

    for (unsigned int cycle = 0; cycle < parameters.n_refinements; ++cycle)
    {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            make_grid();
            setup_dofs();
        }
        else
            refine_mesh();

        newton_iteration(parameters.tolerance,
                         parameters.max_iter,
                         initial_step);


        const Tensor<1,dim> average_boundary_traction
        = compute_boundary_traction();

        std::cout << "   Average traction: " << average_boundary_traction << std::endl;

        output_results(cycle);

        if (cycle == 0)
            initial_step = false;
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::TopographySolver<2>;
template class TopographyProblem::TopographySolver<3>;
