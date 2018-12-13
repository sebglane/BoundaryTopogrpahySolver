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
void TopographySolver<dim>::assemble_system()
{
    TimerOutput::Scope timer_section(computing_timer, "assembly");

    std::cout << "   Assembling system..." << std::endl;

    // reset global objects
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

    const FEValuesExtractors::Scalar    density(0);
    const FEValuesExtractors::Vector    velocity(1);
    const FEValuesExtractors::Scalar    pressure(dim+1);

    FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>              local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // density part
    std::vector<double>         phi_density(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_phi_density(dofs_per_cell);

    // momentum part
    std::vector<double>         div_phi_velocity(dofs_per_cell);
    std::vector<Tensor<1,dim>>  phi_velocity(dofs_per_cell);
    std::vector<Tensor<2,dim>>  grad_phi_velocity(dofs_per_cell);
    std::vector<double>         phi_pressure(dofs_per_cell);

    // start assembly
    for (auto cell: dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        // reset local objects
        local_matrix = 0;
        local_rhs = 0;

        // entropy viscosity density equation
        const double nu_density = 0.1;
        const double nu_velocity = 0.1;

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
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    local_matrix(i, j) += (
                            // continuity equation
                              nu_density * grad_phi_density[j] * grad_phi_density[i]
                            - phi_density[j] * background_velocity_value * grad_phi_density[i]
                            + equation_coefficients[0] * phi_velocity[j] * background_density_gradient * phi_density[i]
                            // incompressibility equation
                            - div_phi_velocity[j] * phi_pressure[i]
                            // momentum equation
                            + nu_velocity * scalar_product(grad_phi_velocity[j], grad_phi_velocity[i])
                            + (grad_phi_velocity[j] * background_velocity_value) * phi_velocity[i]
                            + (background_velocity_gradient * phi_velocity[j]) * phi_velocity[i]
                            - phi_pressure[j] * div_phi_velocity[i]
                            - equation_coefficients[1] * phi_density[j] * gravity_vector * phi_velocity[i]
                           ) * fe_values.JxW(q);
        }
        if (cell->at_boundary())
            for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
                if (cell->face(face_number)->at_boundary() &&
                    cell->face(face_number)->boundary_id() == DomainIdentifiers::TopoBndry)
                {
                    fe_face_values.reinit(cell, face_number);

                    const std::vector<Tensor<1,dim>> normal_vectors = fe_face_values.get_normal_vectors();

                    for (unsigned int q=0; q<n_face_q_points; ++q)
                    {
                        for (unsigned int k=0; k<dofs_per_cell; ++k)
                            // density part
                            phi_density[k]  =   fe_face_values[density].value(k, q);

                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                            for (unsigned int j=0; j<dofs_per_cell; ++j)
                                local_matrix(i, j) +=
                                        // continuity equation
                                          phi_density[j] * normal_vectors[q] * background_velocity_value * phi_density[i]
                                        * fe_face_values.JxW(q);
                    }
                }

        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(local_matrix,
                                               local_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::assemble_system();
template void TopographyProblem::TopographySolver<3>::assemble_system();
