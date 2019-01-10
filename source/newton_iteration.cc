/*
 * newton_iteration.cc
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#include "solver.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::newton_iteration(const double       tolerance,
                                             const unsigned int max_iteration,
                                             const bool         is_initial_step,
                                             const unsigned int level)
{
    double current_res  = 1.0;
    double last_res     = 1.0;
    bool   first_step   = is_initial_step;

    unsigned int iteration = 0;

    while ((first_step || (current_res > tolerance)) &&
                iteration < max_iteration)
    {
        if (first_step)
        {
            // solve problem
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            present_solution = newton_update;
            nonzero_constraints.distribute(present_solution);
            first_step = false;
            // compute residual
            evaluation_point = present_solution;
            assemble_rhs(first_step);
            current_res = system_rhs.l2_norm();
            // output result
            output_results(level, true);
        }
        else
        {
            // solve problem
            evaluation_point = present_solution;
            assemble_system(first_step);
            solve(first_step);
            // line search
            std::cout << "   Line search: " << std::endl;
            for (double alpha = 1.0; alpha > 1e-4; alpha *= 0.5)
            {
                evaluation_point = present_solution;
                evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(evaluation_point);
                assemble_rhs(first_step);
                current_res = system_rhs.l2_norm();
                std::cout << "      alpha = " << std::setw(6)
                          << std::scientific << alpha << std::fixed
                          << std::setw(0)
                          << " residual = " << current_res
                          << std::endl;
                if (current_res < last_res)
                  break;
            }
            present_solution = evaluation_point;
        }
        // output residual
        std::cout << "Iteration: " << iteration
                  << ", residual: "
                  << std::scientific << current_res << std::fixed
                  << std::endl;
        // update residual
        last_res = current_res;
        ++iteration;
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::
TopographySolver<3>::newton_iteration(const double,
                                      const unsigned int,
                                      const bool,
                                      const unsigned int);
