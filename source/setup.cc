/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/function_map.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "solver.h"
#include "equation_data.h"

namespace TopographyProblem {

template<int dim>
void TopographySolver<dim>::setup_dofs()
{
    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    // distribute and renumber block-wise
    dof_handler.distribute_dofs(fe_system);

    DoFRenumbering::block_wise(dof_handler);

    // IO
    std::vector<types::global_dof_index> dofs_per_block(3);
    DoFTools::count_dofs_per_block(dof_handler,
                                   dofs_per_block);

    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "      Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << "      Number of density degrees of freedom: "
              << dofs_per_block[0]
              << std::endl
              << "      Number of velocity degrees of freedom: "
              << dofs_per_block[1]
              << std::endl
              << "      Number of pressure degrees of freedom: "
              << dofs_per_block[2]
              << std::endl;

    // periodicity of grid faces
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
    periodicity_vector;

    switch (dim) {
    case 2:
        GridTools::collect_periodic_faces(dof_handler,
                DomainIdentifiers::Left,
                DomainIdentifiers::Right,
                0,
                periodicity_vector);
        break;
    case 3:
        GridTools::collect_periodic_faces(dof_handler,
                DomainIdentifiers::Left,
                DomainIdentifiers::Right,
                0,
                periodicity_vector);
        GridTools::collect_periodic_faces(dof_handler,
                DomainIdentifiers::Front,
                DomainIdentifiers::Back,
                1,
                periodicity_vector);
        break;
    default:
        Assert(false, ExcImpossibleInDim(dim));
        break;
    }

    // nonzero constraints
    {
        constraints.clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                constraints);

        // periodic boundary conditions for density
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         constraints);

        // constrain normal components of velocity
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert(DomainIdentifiers::BoundaryIds::TopoBndry);

        const EquationData::BackgroundVelocityField<dim>    background_velocity;
        std::map<types::boundary_id, const Function<dim> *> function_map;
        for (const auto it: no_normal_flux_boundaries)
          function_map[it] = &background_velocity;

        VectorTools::compute_nonzero_normal_flux_constraints
        (dof_handler,
         1,
         no_normal_flux_boundaries,
         function_map,
         constraints);

        if (parameters.homogeneous_bc_at_bottom)
        {
            // zero function
            const Functions::ZeroFunction<dim>  zero_function(dim+2);

            // constrain density at bottom
            const FEValuesExtractors::Scalar    density(0);
            VectorTools::interpolate_boundary_values
            (dof_handler,
             DomainIdentifiers::Bottom,
             zero_function,
             constraints,
             fe_system.component_mask(density));

            // constrain velocity at bottom
            const FEValuesExtractors::Vector    velocity(1);
            VectorTools::interpolate_boundary_values
            (dof_handler,
             DomainIdentifiers::Bottom,
             zero_function,
             constraints,
             fe_system.component_mask(velocity));
        }

        constraints.close();
    }
    // stokes matrix and vector setup
    setup_system_matrix(dofs_per_block);
    solution.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
}

template<int dim>
void TopographySolver<dim>::setup_system_matrix
(const std::vector<types::global_dof_index> &dofs_per_block)
{
    system_matrix.clear();

    Table<2,DoFTools::Coupling> coupling;
    coupling.reinit(dim+2, dim+2);

    // density-density coupling
    coupling[0][0] = DoFTools::always;

    // density-velocity coupling
    for (unsigned int c=0; c<dim; ++c)
    {
        coupling[0][c+1] = DoFTools::always;
        coupling[c+1][0] = DoFTools::always;
    }

    // momentum-momentum coupling
    for (unsigned int c=0; c<dim+1; ++c)
        for (unsigned int d=0; d<dim+1; ++d)
            if (c<dim || d<dim)
                coupling[c+1][d+1] = DoFTools::always;
            else if ((c==dim && d<dim) || (c<dim && d==dim))
                coupling[c+1][d+1] = DoFTools::always;
            else
                coupling[c+1][d+1] = DoFTools::none;

    BlockDynamicSparsityPattern dsp(dofs_per_block,
                                    dofs_per_block);

    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    dsp,
                                    constraints);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

}  // namespace TopographyProblem

// explicit instantiation
template void TopographyProblem::TopographySolver<2>::setup_dofs();
template void TopographyProblem::TopographySolver<3>::setup_dofs();

template void TopographyProblem::TopographySolver<2>::setup_system_matrix
(const std::vector<types::global_dof_index> &);
template void TopographyProblem::TopographySolver<3>::setup_system_matrix
(const std::vector<types::global_dof_index> &);
