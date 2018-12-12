/*
 * buoyant_fluid_solver.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>

//#include "assembly_data.h"
#include "parameters.h"

namespace TopographyProblem {

using namespace dealii;

/*
 *
 *
 */
template <int dim>
class TopographySolver
{
public:
    TopographySolver(Parameters &parameters_);

    void run();

private:
    void make_grid();

    void setup_dofs();

    void setup_system_matrix(const std::vector<types::global_dof_index> &dofs_per_block);

    void assemble(const bool initial_step,
                  const bool assemble_matrix);

    void assemble_system(const bool initial_step);

    void assemble_rhs(const bool initial_step);

    void solve(const bool initial_step);

    void newton_iteration(const double tolerance,
                          const unsigned int max_iteration,
                          const bool is_initial_step);

    void output_results(const unsigned int level = 0) const;

    void refine_mesh();

    Parameters                 &parameters;

    std::vector<double>         equation_coefficients;

    const Tensor<1,dim>         rotation_vector;
    const Tensor<1,dim>         gravity_vector;
    const Tensor<1,dim>         background_velocity_value;
    const Tensor<1,dim>         background_field_value;
    const Tensor<1,dim>         background_density_gradient;
    const Tensor<2,dim>         background_velocity_gradient;
    const Tensor<2,dim>         background_field_gradient;

    Triangulation<dim>          triangulation;

    // finite elements and dof handler
    FESystem<dim>               fe_system;
    DoFHandler<dim>             dof_handler;

    // constraints
    ConstraintMatrix            nonzero_constraints;
    ConstraintMatrix            zero_constraints;

    // system matrix
    BlockSparsityPattern        sparsity_pattern;
    BlockSparseMatrix<double>   system_matrix;

    // vectors
    BlockVector<double>         evaluation_point;
    BlockVector<double>         present_solution;
    BlockVector<double>         newton_update;
    BlockVector<double>         system_rhs;

    // monitor of computing times
    TimerOutput                 computing_timer;

private:
    /*
     *
    // working stream methods
    void local_assemble_matrix(
            const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::Matrix<dim>  &scratch,
            Assembly::CopyData::Matrix<dim> &data);
    void copy_local_to_global_matrix(
            const Assembly::CopyData::Matrix<dim> &data);

    void local_assemble_rhs(
            const typename hp::DoFHandler<dim>::active_cell_iterator &cell,
            Assembly::Scratch::RightHandSide<dim>   &scratch,
            Assembly::CopyData::RightHandSide<dim> &data);
    void copy_local_to_global_rhs(
            const Assembly::CopyData::RightHandSide<dim> &data);
     *
     */

};

}  // namespace BouyantFluid

#endif /* INCLUDE_SOLVER_H_ */
