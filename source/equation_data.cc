/*
 * equation_data.cc
 *
 *  Created on: Dec 7, 2018
 *      Author: sg
 */

#include <deal.II/lac/vector.h>

#include "equation_data.h"

namespace EquationData {


template<int dim>
BackgroundVelocityField<dim>::BackgroundVelocityField()
:
Function<dim>(dim)
{
    direction_vector[dim-1] = 1.0;
}

template<int dim>
void BackgroundVelocityField<dim>::vector_value(const Point<dim>    &/* point */,
                                                Vector<double>      &value) const
{
    Assert(value.size() == this->n_components,
           ExcDimensionMismatch(this->n_components, value.size()));
    for (unsigned int d=0; d<this->n_components; ++d)
        value[d] = direction_vector[d];
}


//template<int dim>
//void BackgroundVelocityField<dim>::value_list(const std::vector<Point<dim>>    &points,
//                                              std::vector<Tensor<1,dim>> &values) const
//{
//    Assert(points.size() == values.size(),
//           ExcDimensionMismatch(points.size(), values.size()));
//    for (unsigned int i=0; i<points.size(); ++i)
//        values[i] = this->value(points[i]);
//}

}  // namespace EquationData

// explicit instantiation
template class EquationData::BackgroundVelocityField<2>;
template class EquationData::BackgroundVelocityField<3>;
