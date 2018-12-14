/*
 * entropy_viscosity.cc
 *
 *  Created on: Dec 13, 2018
 *      Author: sg
 */

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include "equation_data.h"
#include "solver.h"

namespace TopographyProblem {

template<int dim>
std::vector<std::pair<double,double>> TopographySolver<dim>::get_range() const
{
  const QIterated<dim>  quadrature_formula(QTrapez<1>(),
                                           parameters.velocity_degree);
  FEValues<dim>         fe_values(fe_system,
                                  quadrature_formula,
                                  update_values);

  const unsigned int    n_q_points = quadrature_formula.size();

  const FEValuesExtractors::Scalar      density(0);
  const FEValuesExtractors::Vector      velocity(1);

  std::vector<double>           density_values(n_q_points);
  std::vector<Tensor<1,dim>>    velocity_values(n_q_points);

  double min_density = std::numeric_limits<double>::max(),
         max_density = -std::numeric_limits<double>::max(),
         min_velocity = std::numeric_limits<double>::max(),
         max_velocity = -std::numeric_limits<double>::max();

  for (auto cell: dof_handler.active_cell_iterators())
  {
      fe_values.reinit (cell);

      fe_values[density].get_function_values(solution,
                                             density_values);
      fe_values[velocity].get_function_values(solution,
                                              velocity_values);

      for (unsigned int q=0; q<n_q_points; ++q)
      {
          min_density = std::min(min_density, density_values[q]);
          max_density = std::max(max_density, density_values[q]);

          min_velocity = std::min(min_velocity, velocity_values[q].norm());
          max_velocity = std::max(max_velocity, velocity_values[q].norm());
      }
  }

  std::vector<std::pair<double,double>> range_values;
  range_values.push_back(std::make_pair(min_density, max_density));
  range_values.push_back(std::make_pair(min_velocity, max_velocity));

  return range_values;
}


template<int dim>
std::pair<double,double>
TopographySolver<dim>::get_entropy_variation(const double average_density,
                                             const double average_velocity) const
{
    const QGauss<dim>   quadrature_formula(parameters.density_degree + 1);

    FEValues<dim>       fe_values(fe_system,
                                  quadrature_formula,
                                  update_values|
                                  update_JxW_values);

    const unsigned int  n_q_points = quadrature_formula.size();

    const FEValuesExtractors::Scalar    density(0);
    const FEValuesExtractors::Vector    velocity(1);

    std::vector<double> density_values(n_q_points);
    std::vector<Tensor<1,dim>> velocity_values(n_q_points);

    double min_entropy_density = std::numeric_limits<double>::max(),
           max_entropy_density = -std::numeric_limits<double>::max(),
           min_entropy_velocity = std::numeric_limits<double>::max(),
           max_entropy_velocity = -std::numeric_limits<double>::max(),
           area = 0,
           integrated_entropy_density = 0,
           integrated_entropy_velocity = 0;

    for (auto cell: dof_handler.active_cell_iterators())
    {
        fe_values.reinit (cell);
        fe_values[density].get_function_values(solution,
                                               density_values);
        fe_values[velocity].get_function_values(solution,
                                                velocity_values);

        for (unsigned int q=0; q<n_q_points; ++q)
        {
            const double entropy_density = (density_values[q] - average_density) *
                                           (density_values[q] - average_density);
            min_entropy_density = std::min(min_entropy_density, entropy_density);
            max_entropy_density = std::max(max_entropy_density, entropy_density);

            const double entropy_velocity = (velocity_values[q] * velocity_values[q]) - average_velocity * average_velocity;
            min_entropy_velocity = std::min(min_entropy_velocity, entropy_velocity);
            max_entropy_velocity = std::max(max_entropy_velocity, entropy_velocity);

            area += fe_values.JxW(q);
            integrated_entropy_density += entropy_density * fe_values.JxW(q);
        }
    }
    const double average_entropy_density = integrated_entropy_density / area;
    const double entropy_diff_density = std::max(max_entropy_density - average_entropy_density,
                                                 average_entropy_density - min_entropy_density);

    const double average_entropy_velocity = integrated_entropy_velocity / area;
    const double entropy_diff_velocity = std::max(max_entropy_velocity - average_entropy_velocity,
                                                  average_entropy_velocity - min_entropy_velocity);

    return std::make_pair(entropy_diff_density, entropy_diff_velocity);
}


template<int dim>
double TopographySolver<dim>::compute_density_viscosity(
        const std::vector<double>          &density_values,
        const std::vector<Tensor<1,dim>>   &density_gradients,
        const std::vector<Tensor<1,dim>>   &velocity_values,
        const std::vector<Tensor<2,dim>>   &velocity_gradients,
        const double                        average_density,
        const double                        global_entropy_variation,
        const double                        cell_diameter) const
{
    const unsigned int n_q_points = density_values.size();

    double max_residual = 0;
    double max_velocity = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
    {
        const double        density     = density_values[q];
        const Tensor<1,dim> velocity    = velocity_values[q] + background_velocity_value;
        const double        div_velocity= std::abs(trace(velocity_gradients[q]));
        const double        v_grad_rho  = velocity * density_gradients[q];

        double residual = std::abs(v_grad_rho) * std::abs(density - average_density)
                        + 0.5  * std::pow((density - average_density), 2) * div_velocity;
        max_residual = std::max (residual, max_residual);
        max_velocity = std::max (std::sqrt(velocity*velocity), max_velocity);
    }
    const double max_viscosity = parameters.c_max * cell_diameter * max_velocity;

    const double entropy_viscosity =
            parameters.c_entropy * cell_diameter * cell_diameter *
            max_residual / global_entropy_variation;

    if (entropy_viscosity > 0.0)
        return std::min(max_viscosity, entropy_viscosity);
    else if (max_viscosity > 0.0)
        return max_viscosity;
    else
        return parameters.default_viscosity;
}

template<int dim>
double TopographySolver<dim>::compute_velocity_viscosity(
        const std::vector<double>          &density_values,
        const std::vector<Tensor<1,dim>>   &density_gradients,
        const std::vector<Tensor<1,dim>>   &velocity_values,
        const std::vector<Tensor<2,dim>>   &velocity_gradients,
        const double                        average_density,
        const double                        global_entropy_variation,
        const double                        cell_diameter) const
{
    const unsigned int n_q_points = velocity_values.size();

    double max_density = 0;
    double max_velocity = 0;

    double max_density_residual = 0;
    double max_velocity_residual = 0;

    for (unsigned int q=0; q < n_q_points; ++q)
    {
        const double        density     = density_values[q];
        const Tensor<1,dim> velocity    = velocity_values[q];
        const Tensor<2,dim> grad_velocity = velocity_gradients[q];
        const double        div_velocity= std::abs(trace(grad_velocity));
        const double        v_grad_rho  = (velocity + background_velocity_value) * density_gradients[q];

        double density_residual = std::abs(v_grad_rho) * std::abs(density - average_density)
                                + 0.5  * std::pow((density - average_density), 2) * div_velocity;
        double velocity_residual = std::abs(velocity * background_velocity_gradient * velocity)
                                 + std::abs(background_velocity_value * grad_velocity * velocity)
                                 + std::abs(background_velocity_value * velocity * div_velocity);

        max_density_residual = std::max(max_density_residual, density_residual / std::abs(density));
        max_velocity_residual = std::max(max_velocity_residual, velocity_residual);

        max_density = std::max(density, max_density);
        max_velocity= std::max(std::sqrt(velocity*velocity), max_velocity);
    }

    const double max_residual = std::max(max_density_residual,
                                         max_velocity_residual);
    const double entropy_viscosity =
            parameters.c_entropy * cell_diameter * cell_diameter * max_density *
            max_residual / global_entropy_variation;

    const double max_viscosity = parameters.c_max * cell_diameter * max_velocity;

    if (entropy_viscosity > 0.0)
        return std::min(max_viscosity, entropy_viscosity);
    else if (max_viscosity > 0.0)
        return max_viscosity;
    else
        return parameters.default_viscosity;
}

}  // namespace TopographyProblem


// explicit instantiation
template std::pair<double,double> TopographyProblem::TopographySolver<2>::get_entropy_variation(const double, const double) const;
template std::pair<double,double> TopographyProblem::TopographySolver<3>::get_entropy_variation(const double, const double) const;

template std::vector<std::pair<double,double>> TopographyProblem::TopographySolver<2>::get_range() const;
template std::vector<std::pair<double,double>> TopographyProblem::TopographySolver<3>::get_range() const;

template double TopographyProblem::TopographySolver<2>::compute_density_viscosity(
        const std::vector<double>               &,
        const std::vector<dealii::Tensor<1,2>>  &,
        const std::vector<dealii::Tensor<1,2>>  &,
        const std::vector<dealii::Tensor<2,2>>  &,
        const double                            ,
        const double                            ,
        const double                            ) const;
template double TopographyProblem::TopographySolver<3>::compute_density_viscosity(
        const std::vector<double>               &,
        const std::vector<dealii::Tensor<1,3>>  &,
        const std::vector<dealii::Tensor<1,3>>  &,
        const std::vector<dealii::Tensor<2,3>>  &,
        const double                            ,
        const double                            ,
        const double                            ) const;

template double TopographyProblem::TopographySolver<2>::compute_velocity_viscosity(
        const std::vector<double>               &,
        const std::vector<dealii::Tensor<1,2>>  &,
        const std::vector<dealii::Tensor<1,2>>  &,
        const std::vector<dealii::Tensor<2,2>>  &,
        const double                            ,
        const double                            ,
        const double                            ) const;
template double TopographyProblem::TopographySolver<3>::compute_velocity_viscosity(
        const std::vector<double>               &,
        const std::vector<dealii::Tensor<1,3>>  &,
        const std::vector<dealii::Tensor<1,3>>  &,
        const std::vector<dealii::Tensor<2,3>>  &,
        const double                            ,
        const double                            ,
        const double                            ) const;
