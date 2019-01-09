/*
 * solver.h
 *
 *  Created on: Nov 7, 2018
 *      Author: sg
 */

#ifndef INCLUDE_SOLVER_H_
#define INCLUDE_SOLVER_H_

#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include "assembly_data.h"
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

    void assemble_system(const bool initial_step);
    void assemble_rhs(const bool initial_step);

    void solve(const bool first_step);

    void newton_iteration(const double tolerance,
                          const unsigned int max_iteration,
                          const bool is_initial_step);

    void output_results(const unsigned int level = 0) const;

    void refine_mesh();

    double                      compute_density_viscosity
    (const std::vector<Tensor<1,dim>>   &velocity_values,
     const double                        cell_diameter) const;

    double                      compute_velocity_viscosity
    (const std::vector<Tensor<1,dim>>   &velocity_values,
     const double                        cell_diameter) const;

    Tensor<1,dim>               compute_boundary_traction() const;

    Parameters                 &parameters;

    std::vector<double>         equation_coefficients;

    const Tensor<1,dim>         rotation_vector;
    const Tensor<1,dim>         gravity_vector;

    const Tensor<1,dim>         background_density_gradient;

    const Tensor<1,dim>         background_velocity_value;
    const Tensor<2,dim>         background_velocity_gradient;

    const Tensor<1,dim>         background_field_value;
    const Tensor<1,dim>         background_field_curl;
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
    BlockSparseMatrix<double>   system_matrix_linear_part;

    // vectors
    BlockVector<double>         evaluation_point;
    BlockVector<double>         present_solution;
    BlockVector<double>         newton_update;
    BlockVector<double>         system_rhs;

    // monitor of computing times
    TimerOutput                 computing_timer;

private:
    // assembly flag
    bool assemble_linear_matrix;

    void local_assemble_nonlinear_matrix(
            const typename DoFHandler<dim>::active_cell_iterator   &cell,
            Assembly::NonLinearScratch<dim>                                 &scratch,
            Assembly::CopyData<dim>                                &data);

    void local_assemble_linear_matrix(
            const typename DoFHandler<dim>::active_cell_iterator   &cell,
            Assembly::LinearScratch<dim>                           &scratch,
            Assembly::CopyData<dim>                                &data);

    void local_assemble_rhs(
            const typename DoFHandler<dim>::active_cell_iterator   &cell,
            Assembly::RightHandSideScratch<dim>                    &scratch,
            Assembly::CopyDataRightHandSide<dim>                   &data,
            const bool                                              initial_step);

    void copy_local_to_global_nonlinear_matrix(
            const Assembly::CopyData<dim>  &data,
            const bool                      initial_step);

    void copy_local_to_global_linear_matrix(
            const Assembly::CopyData<dim>  &data,
            const bool                      initial_step);

    void copy_local_to_global_rhs(
            const Assembly::CopyDataRightHandSide<dim> &data,
            const bool                                  initial_step);
};

}  // namespace BouyantFluid

#endif /* INCLUDE_SOLVER_H_ */
