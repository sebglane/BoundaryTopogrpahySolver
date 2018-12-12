/*
 * assembly.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_values.h>

#include "solver.h"

#include "equation_data.h"

namespace TopographyProblem {

template<>
void TopographySolver<3>::assemble(const bool initial_step,
                                   const bool assemble_matrix)
{
    TimerOutput::Scope timer_section(computing_timer, "assembly");

    const unsigned dim = 3;

    // reset
    if (assemble_matrix)
        system_matrix = 0;
    system_rhs = 0;

    QGauss<dim>         quadrature(parameters.velocity_degree + 1);
    QGauss<dim-1>       face_quadrature(parameters.velocity_degree + 1);

    FEValues<dim>       fe_values(fe_system,
                                  quadrature,
                                  update_values|
                                  update_quadrature_points|
                                  update_JxW_values|
                                  update_gradients);
    FEFaceValues<dim>   fe_face_values(fe_system,
                                       face_quadrature,
                                       update_values|
                                       update_normal_vectors|
                                       update_JxW_values);

    const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
    const unsigned int n_q_points    = quadrature.size();
    const unsigned int n_face_q_points = face_quadrature.size();


    const FEValuesExtractors::Scalar density(0);
    const FEValuesExtractors::Vector velocity(1);
    const FEValuesExtractors::Scalar pressure(dim+1);
    const FEValuesExtractors::Vector field(dim+2);
    const FEValuesExtractors::Scalar scalar(2*dim+2);

    FullMatrix<double>  local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>      local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // density part
    std::vector<double>         phi_density(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_phi_density(dofs_per_cell);

    std::vector<double>         present_density_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_density_gradients(n_q_points);

    // momentum part
    std::vector<double>         div_phi_velocity(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_velocity(dofs_per_cell);
    std::vector<Tensor<2,dim>>  grad_phi_velocity(dofs_per_cell);
    std::vector<double>         phi_pressure(dofs_per_cell);

    std::vector<double>         present_pressure_values(n_q_points);
    std::vector<double>         present_face_pressure_values(n_face_q_points);
    std::vector<Tensor<1,dim>>  present_velocity_values(n_q_points);
    std::vector<Tensor<2,dim>>  present_velocity_gradients(n_q_points);

    // magnetic part
    std::vector<double>         div_phi_field(dofs_per_cell);
    std::vector<double>         phi_scalar(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_field(dofs_per_cell);
    std::vector<Tensor<1,dim>>  curl_phi_field(dofs_per_cell);
    std::vector<Tensor<2,dim>>  grad_phi_field(dofs_per_cell);

    std::vector<double>         present_scalar_values(n_q_points);
    std::vector<double>         present_face_scalar_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_field_values(n_q_points);
    std::vector<Tensor<1,dim>>  present_face_field_curls(n_face_q_points);
    std::vector<Tensor<1,dim>>  present_field_curls(n_q_points);
    std::vector<Tensor<2,dim>>  present_field_gradients(n_q_points);

    // start assembly
    for (auto cell: dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);
        local_matrix = 0;

        // density part
        fe_values[pressure].get_function_values(evaluation_point,
                                                present_density_values);
        fe_values[density].get_function_gradients(evaluation_point,
                                                  present_density_gradients);
        // momentum part
        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);
        fe_values[velocity].get_function_values(evaluation_point,
                                                present_velocity_values);
        fe_values[velocity].get_function_gradients(evaluation_point,
                                                   present_velocity_gradients);
        // magnetic part
        fe_values[scalar].get_function_values(evaluation_point,
                                              present_scalar_values);
        fe_values[field].get_function_values(evaluation_point,
                                             present_field_values);
        fe_values[field].get_function_curls(evaluation_point,
                                            present_field_curls);
        fe_values[field].get_function_gradients(evaluation_point,
                                                present_field_gradients);

        // entropy viscosity density equation
        const double nu_density = 0.1;
        const double nu_velocity = 0.1;
//        = compute_viscosity(global_max_velocity,
//                            global_entropy_variation,
//                            cell->diameter());

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
                // magnetic part
                div_phi_field[k]    =   fe_values[field].divergence(k, q);
                phi_field[k]        =   fe_values[field].value(k, q);
                curl_phi_field[k]   =   fe_values[field].curl(k, q);
                grad_phi_field[k]   =   fe_values[field].gradient(k, q);
                phi_scalar[k]       =   fe_values[scalar].value(k, q);
            }

            for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
                if (assemble_matrix)
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                    {
                        local_matrix(i, j) += (
                                // continuity equation
                                  nu_density * grad_phi_density[j] * grad_phi_density[i]
                                - phi_density[j] * background_velocity_value * grad_phi_density[i]
                                - phi_density[j] * present_velocity_values[q] * grad_phi_density[i]
                                - present_density_values[q] * phi_velocity[j] * grad_phi_density[i]
                                - equation_coefficients[0] * phi_velocity[j] * background_density_gradient * phi_density[i]
                                // incompressibility equation
                                - div_phi_velocity[j] * phi_pressure[i]
                                // momentum equation
                                + nu_velocity * scalar_product(grad_phi_velocity[j], grad_phi_velocity[i])
                                + grad_phi_velocity[j] * background_velocity_value * phi_velocity[i]
                                + background_velocity_gradient * phi_velocity[j] * phi_velocity[i]
                                + present_velocity_gradients[q] * phi_velocity[j] * phi_velocity[i]
                                + grad_phi_velocity[j] * present_velocity_values[q] * phi_velocity[i]
                                + equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, phi_velocity[j]) * phi_velocity[i]
                                - phi_pressure[j] * div_phi_velocity[i]
                                - equation_coefficients[2] * phi_density[j] * gravity_vector * phi_velocity[i]
                                + equation_coefficients[3] * ( curl_phi_field[j] * (grad_phi_velocity[i] * background_field_value)
                                                             + background_field_value * (grad_phi_velocity[i] * curl_phi_field[j])
                                                             + present_field_curls[q] * (grad_phi_velocity[i] * curl_phi_field[j])
                                                             + curl_phi_field[j] * (grad_phi_velocity[i] * present_field_curls[q]) )
                                // Coulomb gauge
                                + div_phi_field[j] * phi_scalar[i]
                                // induction equation
                                - curl_phi_field[j] * curl_phi_field[i]
                                + phi_scalar[j]  * div_phi_field[i]
                                + equation_coefficients[4] * ( cross_product_3d(background_velocity_value , curl_phi_field[j]) * phi_field[i]
                                                             + cross_product_3d(phi_velocity[j] , background_field_value) * phi_field[i]
                                                             + cross_product_3d(present_velocity_values[q], curl_phi_field[j]) * phi_field[i]
                                                             + cross_product_3d(phi_velocity[j], present_field_curls[q]) * phi_field[i] )

                               ) * fe_values.JxW(q);
                    }

                const double present_velocity_divergence =  trace(present_velocity_gradients[q]);
                const double present_field_divergence =  trace(present_field_gradients[q]);
                local_rhs(i) += (
                        // continuity equation
                        - nu_density * present_density_gradients[q] * grad_phi_density[i]
                        + present_density_values[q] * background_velocity_value * grad_phi_density[i]
                        + present_density_values[q] * present_density_gradients[q] * grad_phi_density[i]
                        + equation_coefficients[0] * present_velocity_values[q] * background_density_gradient * phi_density[i]
                        // incompressiblity constraint
                        + present_velocity_divergence * phi_pressure[i]
                        // momentum equation
                        - nu_velocity * scalar_product(present_velocity_gradients[q], grad_phi_velocity[i])
                        - present_velocity_gradients[q] * background_velocity_value * phi_velocity[i]
                        - background_velocity_gradient * present_velocity_values[q] * phi_velocity[i]
                        - present_velocity_gradients[q] * present_velocity_values[q] * phi_velocity[i]
                        - equation_coefficients[1] * 2. * cross_product_3d(rotation_vector, present_velocity_values[q]) * phi_velocity[i]
                        + present_pressure_values[q] * div_phi_velocity[i]

                        + equation_coefficients[2] * present_density_values[q] * gravity_vector * phi_velocity[i]
                        - equation_coefficients[3] * ( present_field_curls[q] * grad_phi_velocity[i] * background_field_value
                                                     + background_field_value * grad_phi_velocity[i] * present_field_curls[q]
                                                     + present_field_curls[q] * grad_phi_velocity[i] * present_field_curls[q] )
                        // Coulomb gauge
                        - present_field_divergence * phi_scalar[i]
                        // induction equation
                        + present_field_curls[q] * curl_phi_field[i]
                        - present_scalar_values[q]  * div_phi_field[i]
                        - equation_coefficients[4] * ( cross_product_3d(present_velocity_values[q] , background_field_value) * phi_field[i]
                                                     + cross_product_3d(background_velocity_value , present_field_curls[q]) * phi_field[i]
                                                     + cross_product_3d(present_velocity_values[q], present_field_curls[q]) * phi_field[i] )
                       ) * fe_values.JxW(q);
            }
        }

        if (cell->at_boundary())
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary() &&
                        (cell->face(face_number)->boundary_id() == DomainIdentifiers::FVB))
                {
                    fe_face_values.reinit(cell, face_number);

                    const std::vector<Tensor<1,dim>> normal_vectors = fe_face_values.get_normal_vectors();

                    // magnetic part
                    fe_face_values[pressure].get_function_values(evaluation_point,
                                                                 present_face_pressure_values);
                    fe_face_values[field].get_function_curls(evaluation_point,
                                                             present_face_field_curls);
                    fe_face_values[scalar].get_function_values(evaluation_point,
                                                               present_face_scalar_values);


                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        for (unsigned int k=0; k<dofs_per_cell; ++k)
                        {
                            // momentum part
                            phi_pressure[k]     =   fe_face_values[pressure].value(k, q);
                            phi_velocity[k]     =   fe_face_values[velocity].value(k, q);
                            // magnetic part
                            curl_phi_field[k]   =   fe_face_values[field].curl(k, q);
                            phi_field[k]    =   fe_face_values[field].value(k, q);
                        }

                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                        {
                            local_rhs(i) += equation_coefficients[3] * (
                                    // momentum equation
                                    - present_face_pressure_values[q] * normal_vectors[q] * phi_velocity[i]
                                    + (normal_vectors[q] * background_field_value) * (present_face_field_curls[q] * phi_velocity[i])
                                    + (normal_vectors[q] * present_face_field_curls[q]) * (background_field_value  * phi_velocity[i])
                                    + (normal_vectors[q] * present_face_field_curls[q]) * (present_face_field_curls[q]  * phi_velocity[i])
                                    // induction equation
                                    + present_face_scalar_values[q] * normal_vectors[q] * phi_field[i]
                                    + (initial_step ? cross_product_3d(normal_vectors[q], background_field_value) * phi_field[i] : 0.0)
                                    ) * fe_face_values.JxW(q);

                            if (assemble_matrix)
                                for (unsigned int j=0; j<dofs_per_cell; ++j)
                                    local_matrix(i, j) +=
                                            // momentum equation
                                            + present_face_pressure_values[q] * normal_vectors[q] * phi_velocity[i]
                                            - equation_coefficients[3]
                                            * ( normal_vectors[q] * background_field_value * curl_phi_field[j] * phi_velocity[i]
                                              + normal_vectors[q] * curl_phi_field[j] * background_field_value  * phi_velocity[i]
                                              + normal_vectors[q] * present_face_field_curls[q] * curl_phi_field[j]  * phi_velocity[i]
                                              + normal_vectors[q] * curl_phi_field[j] * present_face_field_curls[q] * phi_velocity[i]
                                            // induction equation
                                            - phi_scalar[j] * normal_vectors[q] * phi_field[i]
                                            - (initial_step ? cross_product_3d(normal_vectors[q], curl_phi_field[j]) * phi_field[i] : 0.0)
                                              ) * fe_face_values.JxW(q);
                        }
                    }

                }
        cell->get_dof_indices(local_dof_indices);

        const ConstraintMatrix &constraints_used = initial_step ? nonzero_constraints : zero_constraints;
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
}

template <int dim>
void TopographySolver<dim>::assemble_system(const bool initial_step)
{
    std::cout << "   Assemble system..." << std::endl;
    assemble(initial_step, true);
}

template <int dim>
void TopographySolver<dim>::assemble_rhs(const bool initial_step)
{
    std::cout << "   Assemble right-hand side..." << std::endl;
    assemble(initial_step, false);
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<3>::assemble(const bool, const bool);

template void TopographyProblem::TopographySolver<3>::assemble_system(const bool);

template void TopographyProblem::TopographySolver<3>::assemble_rhs(const bool);
