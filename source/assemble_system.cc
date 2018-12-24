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
    double max_nu_density = -std::numeric_limits<double>::max();
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
                                  update_gradients|
                                  update_JxW_values);
    FEFaceValues<dim>   fe_face_values(fe_system,
                                       face_quadrature,
                                       update_values|
                                       update_normal_vectors|
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    const unsigned int n_face_q_points = face_quadrature.size();

    const FEValuesExtractors::Scalar    density(0);
    const FEValuesExtractors::Vector    velocity(1);
    const FEValuesExtractors::Scalar    pressure(dim+1);

    FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>              local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // density part
    std::vector<double>         phi_density(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_phi_density(dofs_per_cell);

    std::vector<double>         present_density_values(n_q_points);
    std::vector<double>         present_face_density_values(n_face_q_points);
    std::vector<Tensor<1,dim>>  present_density_gradients(n_q_points);

    // momentum part
    std::vector<double>         div_phi_velocity(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_velocity(dofs_per_cell);
    std::vector<Tensor<2,dim>>  grad_phi_velocity(dofs_per_cell);
    std::vector<double>         phi_pressure(dofs_per_cell);

    std::vector<double>         present_velocity_divergences(n_q_points);
    std::vector<double>         present_pressure_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_velocity_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_face_velocity_values(n_face_q_points);
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
        // density equation
        fe_values[density].get_function_values(evaluation_point,
                                               present_density_values);
        fe_values[density].get_function_gradients(evaluation_point,
                                                  present_density_gradients);
        // momentum equation
        fe_values[velocity].get_function_divergences(evaluation_point,
                                                     present_velocity_divergences);
        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);

        fe_values[velocity].get_function_gradients(evaluation_point,
                                                   present_velocity_gradients);
        fe_values[velocity].get_function_values(evaluation_point,
                                                present_velocity_values);

        // viscosity density equation
        const double nu_density = compute_density_viscosity(present_velocity_values,
                                                            cell->diameter());
        max_nu_density = std::max(nu_density, max_nu_density);

        // entropy velocity equation
        const double nu_velocity = compute_velocity_viscosity(present_velocity_values,
                                                              cell->diameter());
        max_nu_velocity = std::max(nu_velocity, max_nu_velocity);


        for (unsigned int q=0; q<n_q_points; ++q)
        {
            for (unsigned int k=0; k<dofs_per_cell; ++k)
            {
                // density part
                phi_density[k]      =   fe_values[density].value(k, q);
                grad_phi_density[k] =   fe_values[density].gradient(k, q);
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
                                // continuity equation
                                  nu_density * grad_phi_density[j] * grad_phi_density[i]
                                - phi_density[j] * background_velocity_value * grad_phi_density[i]
                                - phi_density[j] * present_velocity_values[q] * grad_phi_density[i]
                                - present_density_values[q] * phi_velocity[j] * grad_phi_density[i]
                                + equation_coefficients[0] * phi_velocity[j] * background_density_gradient * phi_density[i]
                                // incompressibility equation
                                - div_phi_velocity[j] * phi_pressure[i]
                                // momentum equation
                                + nu_velocity * scalar_product(grad_phi_velocity[j], grad_phi_velocity[i])
                                + (grad_phi_velocity[j] * background_velocity_value) * phi_velocity[i]
                                + (background_velocity_gradient * phi_velocity[j]) * phi_velocity[i]
                                + (grad_phi_velocity[j] * present_velocity_values[q]) * phi_velocity[i]
                                + (present_velocity_gradients[q] * phi_velocity[j]) * phi_velocity[i]
                                + (dim == 3?
                                   equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, phi_velocity[j]) * phi_velocity[i]
                                   : 0.)
                                - phi_pressure[j] * div_phi_velocity[i]
                                - equation_coefficients[2] * phi_density[j] * gravity_vector * phi_velocity[i]
                               ) * fe_values.JxW(q);
                local_rhs(i) += (
                        // continuity equation
                        -  nu_density * present_density_gradients[q] * grad_phi_density[i]
                        + present_density_values[q] * background_velocity_value * grad_phi_density[i]
                        + present_density_values[q] * present_velocity_values[q] * grad_phi_density[i]
                        - equation_coefficients[0] * present_velocity_values[q] * background_density_gradient * phi_density[i]
                        // incompressibility equation
                        + present_velocity_divergences[q] * phi_pressure[i]
                        // momentum equation
                        - nu_velocity * scalar_product(present_velocity_gradients[q], grad_phi_velocity[i])
                        - (present_velocity_gradients[q] * background_velocity_value) * phi_velocity[i]
                        - (background_velocity_gradient * present_velocity_values[q]) * phi_velocity[i]
                        - (present_velocity_gradients[q] * present_velocity_values[q]) * phi_velocity[i]
                        - (dim == 3?
                           equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, present_velocity_values[q]) * phi_velocity[i]
                           : 0.)
                        + present_pressure_values[q] * div_phi_velocity[i]
                        + equation_coefficients[2] * present_density_values[q] * gravity_vector * phi_velocity[i]
                        ) * fe_values.JxW(q);
            }
        }
        if (cell->at_boundary())
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary() &&
                    cell->face(face_number)->boundary_id() == DomainIdentifiers::TopoBndry)
                {
                    fe_face_values.reinit(cell, face_number);

                    const std::vector<Tensor<1,dim>> normal_vectors = fe_face_values.get_normal_vectors();

                    fe_face_values[density].get_function_values(evaluation_point,
                                                                present_face_density_values);
                    fe_face_values[velocity].get_function_values(evaluation_point,
                                                                 present_face_velocity_values);

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        for (unsigned int k=0; k<dofs_per_cell; ++k)
                        {
                            // density part
                            phi_density[k]  =   fe_face_values[density].value(k, q);
                            // momentum part
                            phi_velocity[k]  =  fe_face_values[velocity].value(k, q);
                        }
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            if (assemble_matrix)
                                for (unsigned int j=0; j<dofs_per_cell; ++j)
                                    local_matrix(i, j) += (
                                            // continuity equation
                                              phi_density[j] * normal_vectors[q] * background_velocity_value * phi_density[i]
                                            + phi_density[j] * normal_vectors[q] * present_face_velocity_values[q] * phi_density[i]
                                            + present_face_density_values[q] * normal_vectors[q] * phi_velocity[j] * phi_density[i]
                                            ) * fe_face_values.JxW(q);
                            local_rhs(i) += (
                                    // continuity equation
                                    - present_face_density_values[q] * normal_vectors[q] * background_velocity_value * phi_density[i]
                                    - present_face_density_values[q] * normal_vectors[q] * present_face_velocity_values[q] * phi_density[i]
                                    ) * fe_face_values.JxW(q);
                        }
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
        std::cout << "      maximum viscosity (density): "
                  << max_nu_density
                  << std::endl;
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

