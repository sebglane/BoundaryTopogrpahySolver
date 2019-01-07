/*
 * grid_factor.cc
 *
 *  Created on: Nov 21, 2018
 *      Author: sg
 */

#include "equation_data.h"
#include "grid_factory.h"

namespace GridFactory {

template <int dim>
SinusoidalManifold<dim>::SinusoidalManifold(const double    wavenumber,
                                            const double    amplitude,
                                            const bool      single_wave)
:
ChartManifold<dim,dim,dim-1>(),
wavenumber(wavenumber),
amplitude(amplitude),
single_wave(single_wave)
{}

template <int dim>
std::unique_ptr<Manifold<dim,dim>> SinusoidalManifold<dim>::clone() const
{
  return std::make_unique<SinusoidalManifold<dim>>(wavenumber, amplitude);
}

template<int dim>
Point<dim-1> SinusoidalManifold<dim>::pull_back(const Point<dim> &space_point) const
{
    Point<dim-1> chart_point;
    for (unsigned int d=0; d<dim-1; ++d)
        chart_point[d] = space_point[d];
    return chart_point;
}

template<int dim>
Point<dim> SinusoidalManifold<dim>::push_forward(const Point<dim-1> &chart_point) const
{
    Point<dim> space_point;
    space_point[dim-1] = amplitude;
    if (!single_wave)
        for (unsigned int d=0; d<dim-1; ++d)
        {
            space_point[d] = chart_point[d];
            space_point[dim-1] *= std::sin(wavenumber * chart_point[d]);
        }
    else
    {
        for (unsigned int d=0; d<dim-1; ++d)
            space_point[d] = chart_point[d];
        space_point[dim-1] *= std::sin(wavenumber * chart_point[0]);
    }
    space_point[dim-1] += 1.0;
    return space_point;
}

template<int dim>
DerivativeForm<1, dim-1, dim> SinusoidalManifold<dim>::push_forward_gradient
(const Point<dim-1> &chart_point) const
{
    DerivativeForm<1,dim-1, dim> gradF;
    switch (dim)
    {
    case 2:
    {
        gradF[0][0] = 1.0;
        gradF[1][0] = amplitude * wavenumber *
                      cos(wavenumber * chart_point[0]);
        break;
    }

    case 3:
    {
        gradF[0][0] = 1.0;
        gradF[0][1] = 0.0;

        gradF[1][0] = 0.0;
        gradF[1][1] = 1.0;

        if (!single_wave)
        {
            gradF[2][0] = amplitude * wavenumber *
                       cos(wavenumber * chart_point[0]) *
                       sin(wavenumber * chart_point[1]);
            gradF[2][1] = amplitude * wavenumber *
                       sin(wavenumber * chart_point[0]) *
                       cos(wavenumber * chart_point[1]);
        }
        else
        {
            gradF[2][0] = amplitude * wavenumber *
                       cos(wavenumber * chart_point[0]);
            gradF[2][1] = 0.0;
        }
        break;
    }

    default:
        Assert(false, ExcNotImplemented());
    }
    return gradF;
}

template<int dim>
TopographyBox<dim>::TopographyBox(const double wavenumber,
                                  const double amplitude,
                                  const bool   include_exterior,
                                  const double exterior_length)
:
include_exterior(include_exterior),
exterior_length(exterior_length),
sinus_manifold(wavenumber, amplitude)
{
    Assert(amplitude < 1.0, ExcLowerRangeType<double>(amplitude,1.0));
}


template<int dim>
void TopographyBox<dim>::create_coarse_mesh(Triangulation<dim> &coarse_grid)
{
    if (!include_exterior)
    {
        GridGenerator::hyper_cube(coarse_grid);

        coarse_grid.set_all_manifold_ids(0);
        coarse_grid.set_all_manifold_ids_on_boundary(0);

        for (auto cell: coarse_grid.active_cell_iterators())
        {
            cell->set_material_id(DomainIdentifiers::MaterialIds::Fluid);

            if (cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (cell->face(f)->at_boundary())
                    {
                        std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[dim-1];

                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        {
                            cell->face(f)->set_boundary_id(DomainIdentifiers::BoundaryIds::TopoBndry);
                            cell->face(f)->set_manifold_id(1);
                            break;
                        }
                    }
        }
        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, sinus_manifold);
    }
    else if (include_exterior)
    {
        const Point<dim> origin;
        Point<dim> corner;
        for (unsigned int d=0; d<dim-1; ++d)
            corner[d] = 1.0;
        corner[dim-1] = exterior_length + 1.0;

        std::vector<std::vector<double>> step_sizes;
        for (unsigned int d=0; d<dim-1; ++d)
            step_sizes.push_back(std::vector<double>(1,1.));

        step_sizes.push_back(std::vector<double>{1.0, exterior_length});

        GridGenerator::subdivided_hyper_rectangle(
                coarse_grid,
                step_sizes,
                origin,
                corner);

        coarse_grid.set_all_manifold_ids(0);
        coarse_grid.set_all_manifold_ids_on_boundary(0);

        for (auto cell: coarse_grid.active_cell_iterators())
        {
            if (cell->center()[dim-1] < 1.0)
                cell->set_material_id(DomainIdentifiers::MaterialIds::Fluid);
            else if (cell->center()[dim-1] > 1.0)
                cell->set_material_id(DomainIdentifiers::MaterialIds::Vacuum);

            if (cell->at_boundary())
                for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    if (!cell->face(f)->at_boundary())
                    {
                        std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[dim-1];

                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        {
                            cell->face(f)->set_manifold_id(1);
                        }
                    }
        }
        interpolation_manifold.initialize(coarse_grid);
        coarse_grid.set_manifold(0, interpolation_manifold);
        coarse_grid.set_manifold(1, sinus_manifold);
    }
    else
        Assert(false, ExcInternalError());

    // assignment of boundary identifiers
    for (auto cell: coarse_grid.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);

                    // x-coordinates
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        coord[v] = cell->face(f)->vertex(v)[0];
                    // left boundary
                    if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                    {
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Left);
                    }
                    // left boundary
                    else if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                    {
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Right);
                    }

                    switch (dim)
                    {
                    case 2:
                        // y-coordinates
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[1];
                        // bottom boundary
                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d) < tol;}))
                            cell->face(f)->set_boundary_id(DomainIdentifiers::Bottom);
                        // top boundary
                        else if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - exterior_length) < tol;}) && include_exterior)
                            cell->face(f)->set_boundary_id(DomainIdentifiers::FVB);
                        break;
                    case 3:
                        // y-coordinates
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[1];
                        // front boundary
                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d) < tol;}))
                            cell->face(f)->set_boundary_id(DomainIdentifiers::Front);
                        // back boundary
                        else if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                            cell->face(f)->set_boundary_id(DomainIdentifiers::Back);

                        // z-coordinates
                        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                            coord[v] = cell->face(f)->vertex(v)[2];
                        // bottom boundary
                        if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d) < tol;}))
                            cell->face(f)->set_boundary_id(DomainIdentifiers::Bottom);
                        // top boundary
                        else if (std::all_of(coord.begin(), coord.end(),
                                [&](double d)->bool{return std::abs(d - exterior_length) < tol;}) &&
                                include_exterior)
                            cell->face(f)->set_boundary_id(DomainIdentifiers::FVB);
                        break;
                    default:
                        Assert(false, ExcImpossibleInDim(dim));
                        break;
                    }
                }
}
}  // namespace GridFactory

// explicit instantiation
template class GridFactory::TopographyBox<2>;
template class GridFactory::TopographyBox<3>;
template class GridFactory::SinusoidalManifold<2>;
template class GridFactory::SinusoidalManifold<3>;
