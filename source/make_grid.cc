/*
 * make_grid.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/grid/grid_out.h>

#include "equation_data.h"
#include "grid_factory.h"
#include "solver.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::make_grid()
{
    TimerOutput::Scope timer_section(computing_timer, "make grid");

    std::cout << "   Making grid..." << std::endl;

    GridFactory::TopographyBox<dim>     topography_mesh(2.*numbers::PI,
                                                        parameters.amplitude / parameters.wave_length);

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
}
}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::make_grid();
template void TopographyProblem::TopographySolver<3>::make_grid();
