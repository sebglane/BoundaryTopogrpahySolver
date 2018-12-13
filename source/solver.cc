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
gravity_vector(-Point<dim>::unit_vector(dim-1)),
background_velocity_value(Point<dim>::unit_vector(0)),
background_density_gradient(-Point<dim>::unit_vector(dim-1)),
background_velocity_gradient(),
// coefficients
equation_coefficients{parameters.S,
                      1. / (parameters.Froude * parameters.Froude)},
// triangulation
triangulation(),
// finite element part
fe_system(FE_Q<dim>(parameters.density_degree), 1,
          FESystem<dim>(FE_Q<dim>(parameters.velocity_degree), dim), 1,
          FE_Q<dim>(parameters.velocity_degree - 1), 1),
dof_handler(triangulation),
// monitor
computing_timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{}

template<int dim>
void TopographySolver<dim>::output_results(const unsigned int level) const
{
    std::cout << "   Output results..." << std::endl;

    PostProcessor<dim>  postprocessor;

    // prepare data out object
    DataOut<dim, DoFHandler<dim>>    data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, postprocessor);

    data_out.build_patches();

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
                                       solution,
                                       estimated_error_per_cell,
                                       fe_system.component_mask(velocity));
    // set refinement flags
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.7, 0.3);

    // preparing temperature solution transfer
    std::vector<BlockVector<double>> x_solution(1);
    x_solution[0] = solution;
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
        tmp_solution[0].reinit(solution);
        solution_transfer.interpolate(x_solution, tmp_solution);

        solution = tmp_solution[0];

        constraints.distribute(solution);
    }
}


template<int dim>
void TopographySolver<dim>::run()
{
    make_grid();

    setup_dofs();

    assemble_system();

    solve();

    output_results();

    refine_mesh();
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::TopographySolver<2>;
template class TopographyProblem::TopographySolver<3>;
