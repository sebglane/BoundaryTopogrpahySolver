/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>

#include "equation_data.h"
#include "grid_factory.h"
#include "solver.h"

namespace TopographyProblem {

template<>
void TopographySolver<2>::make_grid()
{
    const unsigned int dim = 2;

    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    GridFactory::TopographyBox<dim>     topography_mesh(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength);

    topography_mesh.create_coarse_mesh(triangulation);

    const unsigned int n_global_refinements = parameters.n_initial_refinements;
    // initial global refinements
    if (n_global_refinements > 0)
    {
        triangulation.refine_global(n_global_refinements);
        std::cout << "      Number of cells after "
                  << n_global_refinements
                  << " global refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    const unsigned int n_boundary_refinements = parameters.n_boundary_refinements;
    // initial boundary refinements
    if (n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<n_boundary_refinements; ++step)
        {
            for (auto cell: triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                        if (cell->face(f)->boundary_id() == DomainIdentifiers::TopoBndry)
                            cell->set_refine_flag();
            triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells after "
                  << n_boundary_refinements
                  << " interface refinements: "
                  << triangulation.n_active_cells()
                  << std::endl;
    }

    std::ofstream out("grid.vtk");
    GridOut grid_out;
    grid_out.write_vtk(triangulation, out);
    std::cout << "      Grid written to grid.vtk" << std::endl;
    out.close();
}

template<>
void TopographySolver<3>::make_grid()
{
    const unsigned int dim = 3;

    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    Triangulation<dim-1>                plane_triangulation;

    GridFactory::TopographyBox<dim-1>   topography_mesh(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength);

    topography_mesh.create_coarse_mesh(plane_triangulation);

    const unsigned int n_global_refinements = parameters.n_initial_refinements;
    // initial global refinements
    if (n_global_refinements > 0)
    {
        plane_triangulation.refine_global(n_global_refinements);

        std::cout << "      Number of cells in 2D after "
                  << n_global_refinements
                  << " global refinements: "
                  << plane_triangulation.n_active_cells()
                  << std::endl;
    }

    const unsigned int n_boundary_refinements = parameters.n_boundary_refinements;
    // initial boundary refinements
    if (n_boundary_refinements > 0)
    {
        for (unsigned int step=0; step<n_boundary_refinements; ++step)
        {
            for (auto cell: plane_triangulation.active_cell_iterators())
                if (cell->at_boundary())
                    for (unsigned int f=0; f<GeometryInfo<dim-1>::faces_per_cell; ++f)
                        if (cell->face(f)->boundary_id() == DomainIdentifiers::TopoBndry)
                            cell->set_refine_flag();
            plane_triangulation.execute_coarsening_and_refinement();
        }
        std::cout << "      Number of cells in 2D after "
                  << n_boundary_refinements
                  << " interface refinements: "
                  << plane_triangulation.n_active_cells()
                  << std::endl;
    }

    std::ofstream out("plane-grid.vtk");
    GridOut grid_out;
    grid_out.write_vtk(plane_triangulation, out);
    std::cout << "      2D Grid written to plane-grid.vtk" << std::endl;
    out.close();

    plane_triangulation.clear();

    GridIn<dim-1>   grid_in;
    grid_in.attach_triangulation(plane_triangulation);
    std::ifstream   in("plane-grid.vtk");
    grid_in.read_vtk(in);
    std::cout << "      2D Grid read from plane-grid.vtk" << std::endl;
    in.close();

    GridGenerator::extrude_triangulation(plane_triangulation,
                                         3,
                                         0.1,
                                         triangulation);

    triangulation.set_all_manifold_ids(0);
    triangulation.set_all_manifold_ids_on_boundary(0);

    const unsigned int normal_direction = 1;
    const unsigned int wave_direction = 0;

    GridFactory::SinusoidalManifold<dim> sinus_manifold(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wavelength,
                                                        normal_direction,
                                                        true,
                                                        wave_direction);

    const double tol = 1e-12;
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                {
                    std::vector<double> coord(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                    {
                        const Point<dim>    vertex = cell->face(f)->vertex(v);
                        const Point<dim>    space_point = sinus_manifold.push_forward(sinus_manifold.pull_back(vertex));
                        coord[v] = vertex[normal_direction] - space_point[normal_direction];
                    }
                    if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_manifold_id(1);
                }

    TransfiniteInterpolationManifold<dim>   interpolation_manifold;
    interpolation_manifold.initialize(triangulation);

    triangulation.set_manifold(0, interpolation_manifold);
    triangulation.set_manifold(1, sinus_manifold);

    // assignment of boundary identifiers
    for (auto cell: triangulation.active_cell_iterators())
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
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Left);
                    // right boundary
                    else if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d - 1.0) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Right);

                    // y-coordinates
                    const double height = 1.0;
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        coord[v] = cell->face(f)->vertex(v)[1];
                    // bottom boundary
                    if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Bottom);
                    // top boundary
                    else if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d - height) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::TopoBndry);

                    // z-coordinates
                    const double depth = 0.1;
                    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_face; ++v)
                        coord[v] = cell->face(f)->vertex(v)[dim-1];
                    // front boundary
                    if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Front);
                    // back boundary
                    else if (std::all_of(coord.begin(), coord.end(),
                            [&](double d)->bool{return std::abs(d - depth) < tol;}))
                        cell->face(f)->set_boundary_id(DomainIdentifiers::Back);
                }

    std::set<types::boundary_id>    boundary_ids;
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->at_boundary())
            for (unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                    boundary_ids.insert(cell->face(f)->boundary_id());

    out.open("grid.vtk");
    grid_out.write_vtk(triangulation, out);
    std::cout << "      Grid written to grid.vtk" << std::endl;
    out.close();
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::make_grid();
template void TopographyProblem::TopographySolver<3>::make_grid();
