/*
 * linear_solve.cc
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>

#include "solver.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::solve()
{
    std::cout << "   Solving linear system..." << std::endl;

    TimerOutput::Scope timer_section(computing_timer, "linear solve");

    SparseDirectUMFPACK     direct_solver;
    direct_solver.solve(system_matrix, system_rhs);

    solution = system_rhs;
    constraints.distribute(solution);

    const double mean_pressure = VectorTools::compute_mean_value(dof_handler,
                                                                 QGauss<dim>(parameters.velocity_degree + 1),
                                                                 solution,
                                                                 dim+1);
    solution.block(2).add(-mean_pressure);
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::solve();
template void TopographyProblem::TopographySolver<3>::solve();
