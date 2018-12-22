/*
 * assemble_system.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include "solver.h"

#include "equation_data.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::assemble(const bool initial_step, const bool assemble_matrix)
{
    TimerOutput::Scope timer_section(computing_timer, "assembly");

    // maximum viscosities
    double max_nu_velocity = -std::numeric_limits<double>::max();

    // reset global objects
    if (assemble_matrix)
        system_matrix = 0;
    system_rhs = 0;

    QGauss<dim>         quadrature(parameters.velocity_degree + 1);
    QGauss<dim-1>       face_quadrature(parameters.velocity_degree + 1);

    FEValues<dim>       fe_values(fe_system,
                                  quadrature,
                                  update_values|
                                  update_JxW_values|
                                  update_gradients);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();

    const FEValuesExtractors::Vector    velocity(0);
    const FEValuesExtractors::Scalar    pressure(dim);

    FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>              local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // momentum part
    std::vector<double>         div_phi_velocity(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_velocity(dofs_per_cell);
    std::vector<Tensor<2,dim>>  grad_phi_velocity(dofs_per_cell);
    std::vector<double>         phi_pressure(dofs_per_cell);

    std::vector<double>         present_pressure_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_velocity_values(n_q_points);
    std::vector<Tensor<2,dim>>  present_velocity_gradients(n_q_points);

    // start assembly
    for (auto cell: dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        // reset local objects
        if (assemble_matrix)
            local_matrix = 0;
        local_rhs = 0;

        // compute present values for nonlinearity
        // momentum equation
        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);

        fe_values[velocity].get_function_gradients(evaluation_point,
                                                   present_velocity_gradients);
        fe_values[velocity].get_function_values(evaluation_point,
                                                present_velocity_values);

        // entropy velocity equation
        const double nu_velocity = 0.1;// compute_velocity_viscosity(present_velocity_values, cell->diameter());
        max_nu_velocity = std::max(nu_velocity, max_nu_velocity);


        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                // momentum part
                div_phi_velocity[k] =   fe_values[velocity].divergence(k, q);
                phi_velocity[k]     =   fe_values[velocity].value(k, q);
                grad_phi_velocity[k]=   fe_values[velocity].gradient(k, q);
                phi_pressure[k]     =   fe_values[pressure].value(k, q);
            }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                if (assemble_matrix)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                        local_matrix(i, j) += (
                                // incompressibility equation
                                - div_phi_velocity[j] * phi_pressure[i]
                                // momentum equation
                                + nu_velocity * scalar_product(grad_phi_velocity[j], grad_phi_velocity[i])
                                + (grad_phi_velocity[j] * background_velocity_value) * phi_velocity[i]
                                + (background_velocity_gradient * phi_velocity[j]) * phi_velocity[i]
                                + (grad_phi_velocity[j] * present_velocity_values[q]) * phi_velocity[i]
                                + (present_velocity_gradients[q] * phi_velocity[j]) * phi_velocity[i]
                                - phi_pressure[j] * div_phi_velocity[i]
                               ) * fe_values.JxW(q);
                const double present_velocity_divergence = trace(present_velocity_gradients[q]);
                local_rhs(i) += (
                        // incompressibility equation
                          present_velocity_divergence * phi_pressure[i]
                        // momentum equation
                        - nu_velocity * scalar_product(present_velocity_gradients[q], grad_phi_velocity[i])
                        - (present_velocity_gradients[q] * background_velocity_value) * phi_velocity[i]
                        - (background_velocity_gradient * present_velocity_values[q]) * phi_velocity[i]
                        - (present_velocity_gradients[q] * present_velocity_values[q]) * phi_velocity[i]
                        + present_pressure_values[q] * div_phi_velocity[i]
                        ) * fe_values.JxW(q);
            }
        }

        cell->get_dof_indices(local_dof_indices);

        const ConstraintMatrix &constraints_used = (initial_step ? nonzero_constraints
                                                                 : zero_constraints);

        if (assemble_matrix)
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);
        else
            constraints_used.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs);
    }
    if (assemble_matrix)
    {
        std::cout << "      maximum viscosity (velocity): "
                  << max_nu_velocity
                  << std::endl;
    }
}

template<int dim>
void TopographySolver<dim>::assemble_system(const bool initial_step)
{
    std::cout << "   Assembling system..." << std::endl;
    assemble(initial_step, true);
}

template<int dim>
void TopographySolver<dim>::assemble_rhs(const bool initial_step)
{
    assemble(initial_step, false);
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::assemble(const bool, const bool);
template void TopographyProblem::TopographySolver<3>::assemble(const bool, const bool);

template void TopographyProblem::TopographySolver<2>::assemble_system(const bool);
template void TopographyProblem::TopographySolver<3>::assemble_system(const bool);

template void TopographyProblem::TopographySolver<2>::assemble_rhs(const bool);
template void TopographyProblem::TopographySolver<3>::assemble_rhs(const bool);

