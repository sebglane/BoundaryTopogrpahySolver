/*
 * assembly_data.h
 *
 *  Created on: Dec 24, 2018
 *      Author: sg
 */

#ifndef INCLUDE_ASSEMBLY_DATA_H_
#define INCLUDE_ASSEMBLY_DATA_H_


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_base.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <vector>

namespace Assembly {

using namespace dealii;

namespace Scratch {

template<int dim>
struct MatrixData
{
    MatrixData(const FiniteElement<dim> &finite_element,
               const Quadrature<dim>    &quadrature,
               const Quadrature<dim-1>  &face_squadrature,
               const UpdateFlags         update_flags,
               const UpdateFlags         face_update_flags);

    MatrixData(const MatrixData<dim>    &scratch);

    FEValues<dim>               fe_values;
    FEFaceValues<dim>           fe_face_values;

    // density part
    std::vector<double>         phi_density;
    std::vector<Tensor<1,dim>>  grad_phi_density;

    std::vector<double>         present_density_values;
    std::vector<double>         present_face_density_values;
    std::vector<Tensor<1,dim>>  present_density_gradients;

    // momentum part
    std::vector<double>         div_phi_velocity;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<double>         phi_pressure;

    std::vector<double>         present_velocity_divergences;
    std::vector<double>         present_pressure_values;
    std::vector<Tensor<1,dim>>  present_velocity_values;
    std::vector<Tensor<1,dim>>  present_face_velocity_values;
    std::vector<Tensor<2,dim>>  present_velocity_gradients;
};


template <int dim>
struct RightHandSideData
{
    RightHandSideData(const FiniteElement<dim> &finite_element,
                      const Quadrature<dim>    &quadrature,
                      const Quadrature<dim-1>  &face_squadrature,
                      const UpdateFlags         update_flags,
                      const UpdateFlags         face_update_flags);

    RightHandSideData(const RightHandSideData<dim>  &scratch);

    FEValues<dim>               fe_values;
    FEFaceValues<dim>           fe_face_values;

    // density part
    std::vector<double>         phi_density;
    std::vector<Tensor<1,dim>>  grad_phi_density;

    std::vector<double>         present_density_values;
    std::vector<double>         present_face_density_values;
    std::vector<Tensor<1,dim>>  present_density_gradients;

    // momentum part
    std::vector<double>         div_phi_velocity;
    std::vector<Tensor<1,dim>>  phi_velocity;
    std::vector<Tensor<2,dim>>  grad_phi_velocity;
    std::vector<double>         phi_pressure;

    std::vector<double>         present_velocity_divergences;
    std::vector<double>         present_pressure_values;
    std::vector<Tensor<1,dim>>  present_velocity_values;
    std::vector<Tensor<1,dim>>  present_face_velocity_values;
    std::vector<Tensor<2,dim>>  present_velocity_gradients;
};

}  // namespace Scratch

namespace Copy {

template <int dim>
struct MatrixData
{
    MatrixData(const FiniteElement<dim>     &finite_element);
    MatrixData(const MatrixData<dim>        &data);

    FullMatrix<double>                      local_matrix;
    Vector<double>                          local_rhs;

    std::vector<types::global_dof_index>    local_dof_indices;
};

template <int dim>
struct RightHandSideData
{
    RightHandSideData(const FiniteElement<dim>      &finite_element);
    RightHandSideData(const RightHandSideData<dim>  &data);

    FullMatrix<double>                      matrix_for_bc;
    Vector<double>                          local_rhs;

    std::vector<types::global_dof_index>    local_dof_indices;
};

}  // namespace Copy

} // namespace Assembly


#endif /* INCLUDE_ASSEMBLY_DATA_H_ */
