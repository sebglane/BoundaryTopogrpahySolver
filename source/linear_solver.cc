/*
 * linear_solve.cc
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#include <deal.II/lac/sparse_direct.h>

#include "solver.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::solve(const bool initial_step)
{
    std::cout << "   Solving linear system..." << std::endl;

    const ConstraintMatrix  &constraints_used
        = initial_step ? nonzero_constraints : zero_constraints;

    SparseDirectUMFPACK     direct_solver;
    direct_solver.solve(system_matrix, system_rhs);

    newton_update = system_rhs;
    constraints_used.distribute(newton_update);
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<3>::solve(const bool);
