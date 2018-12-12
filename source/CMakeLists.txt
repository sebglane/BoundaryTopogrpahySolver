# Set the name of the project and target:
SET(TARGET "main")

SET(TARGET_SRC
  main.cc
  assemble_system.cc
  equation_data.cc
  grid_factory.cc
  linear_solver.cc
  make_grid.cc
  newton_iteration.cc
  parameters.cc
  postprocessor.cc
  setup.cc
  solver.cc
  # You can specify additional files here!
)


PROJECT(${TARGET} CXX)

INCLUDE_DIRECTORIES(
  include
  ${CMAKE_SOURCE_DIR}/include
  )

DEAL_II_INVOKE_AUTOPILOT()
