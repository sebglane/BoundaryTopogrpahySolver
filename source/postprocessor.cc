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
DataPostprocessor<dim>()
{}

template<>
std::vector<std::string> PostProcessor<3>::get_names() const
{
    const unsigned int dim = 3;

    std::vector<std::string> solution_names;
    // density
    solution_names.push_back("density");
    // velocity
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("velocity");
    // pressure
    solution_names.push_back("pressure");
    // magnetic vector potential
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("vector_potential");
    // magnetic scalar potential
    solution_names.push_back("scalar_potential");
    // magnetic field
    for (unsigned int d=0; d<dim; ++d)
        solution_names.push_back("magnetic_field");

    return solution_names;
}

template<int dim>
UpdateFlags PostProcessor<dim>::get_needed_update_flags() const
{
    return update_values|update_gradients;
}

template<>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
PostProcessor<3>::get_data_component_interpretation() const
{
    const unsigned int dim = 3;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation;

    // density
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // velocity
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // pressure
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // magnetic vector potential
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    // magnetic scalar potential
    component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // magnetic field
    for (unsigned int d=0; d<dim; ++d)
        component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

    return component_interpretation;
}

template <>
void PostProcessor<3>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<3> &inputs,
        std::vector<Vector<double>>              &computed_quantities) const
{
    const unsigned int dim = 3;

    const unsigned int n_quadrature_points = inputs.solution_values.size();

    AssertDimension(computed_quantities.size(),
                    n_quadrature_points);
    AssertDimension(inputs.solution_values[0].size(),
                    2*dim+3);

    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
        AssertDimension(computed_quantities[q].size(), 3*dim+3);
        // density
        computed_quantities[q][0] = inputs.solution_values[q][0];
        // velocity
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+1] = inputs.solution_values[q][d+1];
        // pressure
        computed_quantities[q][dim+1] = inputs.solution_values[q][dim+1];
        // magnetic vector potential
        for (unsigned int d=0; d<dim; ++d)
            computed_quantities[q][d+dim+2] = inputs.solution_values[q][d+dim+2];
        // magnetic scalar potential
        computed_quantities[q][2*dim+2] = inputs.solution_values[q][dim+2];
        // magnetic field
        computed_quantities[q][2*dim+2+1] = inputs.solution_gradients[q][2][1]
                                           -inputs.solution_gradients[q][1][2];
        computed_quantities[q][2*dim+2+2] = inputs.solution_gradients[q][0][2]
                                           -inputs.solution_gradients[q][2][0];
        computed_quantities[q][2*dim+2+3] = inputs.solution_gradients[q][1][0]
                                           -inputs.solution_gradients[q][0][1];
    }
}
}  // namespace TopographyProblem

// explicit instantiation
template class TopographyProblem::PostProcessor<3>;
