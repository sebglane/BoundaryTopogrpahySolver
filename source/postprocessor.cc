/*
 * postprocessor.cc
 *
 *  Created on: Dec 11, 2018
 *      Author: sg
 */

#include "postprocessor.h"

namespace TopographyProblem {

template<int dim>
PostProcessor<dim>::PostProcessor()
:
DataPostprocessor<dim>(),
background_velocity()
{}

template<int dim>
std::vector<std::string> PostProcessor<dim>::get_names() const
{
    std::vector<std::string> solution_names;
    // velocity
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("velocity");
    // pressure
    solution_names.push_back("pressure");
    // total velocity
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("total_velocity");

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values;
}

template<int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<dim>::get_data_component_interpretation() const
{
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation;

    // velocity
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // pressure
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // total velocity
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

    return component_interpretation;
}

template<int dim>
void PostProcessor<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>  &inputs,
        std::vector<Vector<double>>                 &computed_quantities) const
{

    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Vector<double>      background_velocity_values(background_velocity.n_components);

    AssertDimension(computed_quantities.size(),
                    n_quadrature_points);
    AssertDimension(inputs.solution_values[0].size(),
                    dim+1);

    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        AssertDimension(computed_quantities[q].size(), 2*dim+1);
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d] = inputs.solution_values[q][d];
        // pressure
        computed_quantities[q][dim] = inputs.solution_values[q][dim];
        // total velocity
        background_velocity.vector_value(inputs.evaluation_points[q],
                                         background_velocity_values);
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+dim+1] = inputs.solution_values[q][d]
                                            + background_velocity_values[d]  ;
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::PostProcessor<2>;
template class TopographyProblem::PostProcessor<3>;
