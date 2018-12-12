/*
 * setup.cc
 *
 *  Created on: Nov 23, 2018
 *      Author: sg
 */

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/function_map.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/vector_tools.h>

#include "solver.h"
#include "equation_data.h"

namespace TopographyProblem {

template<>
void TopographySolver<3>::setup_dofs()
{
    TimerOutput::Scope timer_section(computing_timer, "setup dofs");

    std::cout << "   Setup dofs..." << std::endl;

    const unsigned int dim = 3;

    // distribute and renumber block-wise
    dof_handler.distribute_dofs(fe_system);

    DoFRenumbering::block_wise(dof_handler);

    // IO
    std::vector<types::global_dof_index> dofs_per_block(5);
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
              << std::endl
              << "      Number of magnetic degrees of freedom: "
              << dofs_per_block[3]
              << std::endl
              << "      Number of magnetic degrees of freedom: "
              << dofs_per_block[4]
              << std::endl;

    // periodicity of grid faces
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
    periodicity_vector;

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

    // nonzero constraints
    {
        nonzero_constraints.clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                nonzero_constraints);

        // periodic boundary conditions for density
        const FEValuesExtractors::Scalar    density(0);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         nonzero_constraints,
         fe_system.component_mask(density));

        // periodic boundary conditions for velocity
        const FEValuesExtractors::Vector    velocity(1);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         nonzero_constraints,
         fe_system.component_mask(velocity));

        // periodic boundary conditions for pressure
        const FEValuesExtractors::Scalar    pressure(dim+1);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         nonzero_constraints,
         fe_system.component_mask(pressure));

        // periodic boundary conditions for magnetic field
        const FEValuesExtractors::Vector    magnetic_field(dim+2);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         nonzero_constraints,
         fe_system.component_mask(magnetic_field));

        // periodic boundary conditions for scalar field
        const FEValuesExtractors::Scalar    scalar_field(2*dim+2);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         nonzero_constraints,
         fe_system.component_mask(scalar_field));

        // constrain normal components of velocity
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert(0);

        const EquationData::BackgroundVelocityField<dim>    background_velocity;
        std::map<types::boundary_id, const Function<dim> *> function_map;
        for (const auto it: no_normal_flux_boundaries)
          function_map[it] = &background_velocity;

        VectorTools::compute_nonzero_normal_flux_constraints
        (dof_handler,
         1,
         no_normal_flux_boundaries,
         function_map,
         nonzero_constraints);

        // zero function
        const Functions::ZeroFunction<dim>  zero_function(2*dim+3);

        // constrain density at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         nonzero_constraints,
         fe_system.component_mask(density));

        // constrain velocity at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         nonzero_constraints,
         fe_system.component_mask(velocity));

        // constrain magnetic field at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         nonzero_constraints,
         fe_system.component_mask(magnetic_field));

        nonzero_constraints.close();
    }
    // zero constraints
    {
        zero_constraints.clear();

        DoFTools::make_hanging_node_constraints(dof_handler,
                                                zero_constraints);

        // periodic boundary conditions for density
        const FEValuesExtractors::Scalar    density(0);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         zero_constraints,
         fe_system.component_mask(density));

        // periodic boundary conditions for velocity
        const FEValuesExtractors::Vector    velocity(1);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         zero_constraints,
         fe_system.component_mask(velocity));

        // periodic boundary conditions for pressure
        const FEValuesExtractors::Scalar    pressure(dim+1);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         zero_constraints,
         fe_system.component_mask(pressure));

        // periodic boundary conditions for magnetic field
        const FEValuesExtractors::Vector    magnetic_field(dim+2);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         zero_constraints,
         fe_system.component_mask(magnetic_field));

        // periodic boundary conditions for scalar field
        const FEValuesExtractors::Scalar    scalar_field(2*dim+2);
        DoFTools::make_periodicity_constraints<DoFHandler<dim>>
        (periodicity_vector,
         zero_constraints,
         fe_system.component_mask(scalar_field));

        // constrain normal components of velocity
        std::set<types::boundary_id> no_normal_flux_boundaries;
        no_normal_flux_boundaries.insert(DomainIdentifiers::TopoBndry);
        VectorTools::compute_no_normal_flux_constraints
        (dof_handler, 1,
         no_normal_flux_boundaries,
         zero_constraints);

        // zero function
        const Functions::ZeroFunction<dim>  zero_function(2*dim+3);

        // constrain density at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         zero_constraints,
         fe_system.component_mask(density));

        // constrain velocity at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         zero_constraints,
         fe_system.component_mask(velocity));

        // constrain magnetic field at bottom
        VectorTools::interpolate_boundary_values
        (dof_handler,
         DomainIdentifiers::Bottom,
         zero_function,
         zero_constraints,
         fe_system.component_mask(magnetic_field));

        zero_constraints.close();
    }

    // stokes matrix and vector setup
    setup_system_matrix(dofs_per_block);
    evaluation_point.reinit(dofs_per_block);
    present_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
}

template<int dim>
void TopographySolver<dim>::setup_system_matrix
(const std::vector<types::global_dof_index> &dofs_per_block)
{
    system_matrix.clear();

    Table<2,DoFTools::Coupling> coupling;
    switch (dim)
    {
    case 2:
    {
        coupling.reinit(dim+3, dim+3);

        // density-density coupling
        coupling[0][0] = DoFTools::always;

        // density-velocity coupling
        for (unsigned int c=1; c<dim+1; ++c)
        {
            coupling[0][c] = DoFTools::always;
            coupling[c][0] = DoFTools::always;
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

        // momentum-magnetic coupling
        for (unsigned int c=0; c<dim; ++c)
        {
            coupling[c+1][dim+2] = DoFTools::always;
            coupling[dim+2][c+1] = DoFTools::always;
        }

        // magnetic-magnetic coupling
        coupling[dim+2][dim+2] = DoFTools::always;

        break;
    }
    case 3:
    {
        coupling.reinit(2*dim+3, 2*dim+3);

        // density-density coupling
        coupling[0][0] = DoFTools::always;

        // density-velocity coupling
        for (unsigned int c=1; c<dim+1; ++c)
        {
            coupling[0][c] = DoFTools::always;
            coupling[c][0] = DoFTools::always;
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

        // momentum-magnetic coupling
        for (unsigned int c=0; c<dim; ++c)
            for (unsigned int d=0; d<dim; ++d)
            {
                coupling[c+1][d+dim+2] = DoFTools::always;
                coupling[c+dim+2][d+1] = DoFTools::always;
            }

        // magnetic-magnetic coupling
        for (unsigned int c=0; c<dim+1; ++c)
            for (unsigned int d=0; d<dim+1; ++d)
                if (c<dim || d<dim)
                    coupling[c+dim+2][d+dim+2] = DoFTools::always;
                else if ((c==dim && d<dim) || (c<dim && d==dim))
                    coupling[c+dim+2][d+dim+2] = DoFTools::always;
        break;
    }
    default:
        Assert(false, ExcImpossibleInDim(dim));
        break;
    }

    BlockDynamicSparsityPattern dsp(dofs_per_block,
            dofs_per_block);

    DoFTools::make_sparsity_pattern(
            dof_handler,
            coupling,
            dsp,
            nonzero_constraints);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

}  // namespace TopographyProblem


// explicit instantiation
template void TopographyProblem::TopographySolver<3>::setup_dofs();

template void TopographyProblem::TopographySolver<2>::setup_system_matrix
(const std::vector<types::global_dof_index> &);
template void TopographyProblem::TopographySolver<3>::setup_system_matrix
(const std::vector<types::global_dof_index> &);
