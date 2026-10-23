# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    MeshDeformationCallback(; brep_folder, t_start, dt,
                            rbf_type = :cpc2, scale_factor_support_radius = 1.5)

Deform a [`T8codeMesh`](@ref) during the simulation to match a sequence of CAD
geometries. The `.brep` files in `brep_folder` are used one after another in alphabetical
order: the first deformation is performed at time `t_start`, e.g., after the solution
has converged on the initial geometry, and the following ones every `dt` afterwards,
i.e., the geometry of the `k`-th file is used from time `t_start + (k - 1) * dt` on.
No further deformations are performed once all files have been used.

Each deformation is performed by t8code (`t8_cmesh_deform_to_brep`): the vertices on the
boundary of the mesh are moved onto the new CAD geometry, and the displacements of the
inner vertices are interpolated with radial basis functions (RBF) of type `rbf_type`,
either `:cpc2` (compactly supported, with a support radius of
`scale_factor_support_radius` times the mean distance of a boundary vertex to its
neighbors) or `:tps` (globally supported thin plate splines). Afterwards, the geometry of
the mesh and all geometric data of the solver are recomputed. The nodal values of the
solution are kept unchanged, i.e., the solution moves with the mesh nodes.

The mesh must be created from a Gmsh file using its CAD geometry, i.e., via
`T8codeMesh(mesh_file, ndims; use_cad_geometry = true, ...)`, and all `.brep` files must
have the same topology as the initial geometry (the `.brep` file next to `mesh_file`).
User-defined `mapping`s of the mesh are not reapplied after a deformation. This requires
a t8code build with OpenCASCADE and Eigen support.

The callback adds stops at the deformation times to integrators supporting this, e.g., the
ones of OrdinaryDiffEq.jl. Other integrators perform a deformation in the first time step
reaching its time. Place the callback before the [`StepsizeCallback`](@ref) in the
`CallbackSet` so that the time step is computed on the deformed mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
mutable struct MeshDeformationCallback{RealT <: Real}
    const brep_files::Vector{String} # file prefixes, i.e., without `.brep`
    const t_start::RealT
    const dt::RealT
    const rbf_type::Cint
    const scale_factor_support_radius::Float64
    next_file::Int # index of the next file in `brep_files`
end

function MeshDeformationCallback(; brep_folder, t_start, dt,
                                 rbf_type = :cpc2, scale_factor_support_radius = 1.5)
    if !isdir(brep_folder)
        throw(ArgumentError("`brep_folder` is not a directory (provided `brep_folder = $brep_folder`)"))
    end
    if !(dt > 0)
        throw(ArgumentError("`dt` must be positive (provided `dt = $dt`)"))
    end
    if rbf_type === :cpc2
        rbf_type_t8code = Cint(0)
    elseif rbf_type === :tps
        rbf_type_t8code = Cint(1)
    else
        throw(ArgumentError("`rbf_type` must be `:cpc2` or `:tps` (provided `rbf_type = $rbf_type`)"))
    end

    # `readdir` returns the file names sorted alphabetically.
    brep_files = [first(splitext(file))
                  for file in readdir(brep_folder; join = true)
                  if isfile(file) && lowercase(last(splitext(file))) == ".brep"]
    if isempty(brep_files)
        throw(ArgumentError("no `.brep` files found in `brep_folder = $brep_folder`"))
    end

    t_start, dt = promote(t_start, dt)
    deformation_callback = MeshDeformationCallback(brep_files, t_start, dt,
                                                   rbf_type_t8code,
                                                   Float64(scale_factor_support_radius),
                                                   1)

    # The first one is the condition, the second the affect!
    return DiscreteCallback(deformation_callback, deformation_callback,
                            save_positions = (false, false),
                            initialize = initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:MeshDeformationCallback})
    @nospecialize cb # reduce precompilation time

    deformation_callback = cb.affect!
    print(io, "MeshDeformationCallback(t_start=", deformation_callback.t_start,
          ", dt=", deformation_callback.dt, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:MeshDeformationCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        deformation_callback = cb.affect!

        setup = [
            "brep folder" => dirname(first(deformation_callback.brep_files)),
            "#brep files" => length(deformation_callback.brep_files),
            "t_start" => deformation_callback.t_start,
            "dt" => deformation_callback.dt,
            "RBF type" => deformation_callback.rbf_type == 0 ? "CPC2" : "TPS",
            "support radius factor" => deformation_callback.scale_factor_support_radius
        ]
        summary_box(io, "MeshDeformationCallback", setup)
    end
end

# Time at which the geometry of the `k`-th file is used
@inline function deformation_time(deformation_callback::MeshDeformationCallback, k)
    @unpack t_start, dt = deformation_callback
    return t_start + (k - 1) * dt
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: MeshDeformationCallback}
    deformation_callback = cb.affect!
    @unpack brep_files = deformation_callback

    # Skip the deformations scheduled before the initial time, e.g., when restarting
    # a simulation, such that the `k`-th file always belongs to the same time.
    deformation_callback.next_file = 1
    while (deformation_callback.next_file <= length(brep_files) &&
           deformation_time(deformation_callback, deformation_callback.next_file) < t)
        deformation_callback.next_file += 1
    end
    if deformation_callback.next_file > 1
        @warn "Skipping $(deformation_callback.next_file - 1) mesh deformations scheduled before the initial time t = $t"
    end

    add_deformation_tstop!(integrator, deformation_callback)

    return nothing
end

# Make the integrator stop exactly at the next deformation time if it supports `tstops`.
# Trixi.jl's own explicit integrators do not support them; there, the deformation is
# performed in the first time step reaching the deformation time.
function add_deformation_tstop!(integrator, deformation_callback)
    @unpack brep_files, next_file = deformation_callback
    next_file > length(brep_files) && return nothing

    t_next = deformation_time(deformation_callback, next_file)
    if (integrator.t < t_next < last(integrator.sol.prob.tspan) &&
        applicable(add_tstop!, integrator, t_next))
        add_tstop!(integrator, t_next)
    end

    return nothing
end

# This method is called to determine whether the callback should be activated
function (deformation_callback::MeshDeformationCallback)(u, t, integrator)
    @unpack brep_files, next_file = deformation_callback

    return (next_file <= length(brep_files) &&
            t >= deformation_time(deformation_callback, next_file))
end

# This method is called when the callback is activated
function (deformation_callback::MeshDeformationCallback)(integrator)
    @unpack brep_files, next_file = deformation_callback
    semi = integrator.p

    brep_file = brep_files[next_file]
    mpi_isroot() &&
        @info "Deforming mesh to the geometry of $brep_file.brep at t = $(integrator.t)"

    @trixi_timeit timer() "mesh deformation" begin
        deform_mesh!(semi, brep_file, deformation_callback)
    end

    deformation_callback.next_file += 1
    add_deformation_tstop!(integrator, deformation_callback)

    # The right-hand side changed with the geometry, so FSAL stages must be recomputed.
    derivative_discontinuity!(integrator, true)
    return nothing
end

function deform_mesh!(semi::SemidiscretizationHyperbolic, brep_file,
                      deformation_callback)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    deform_mesh!(mesh, equations, solver, cache, brep_file, deformation_callback)

    # The boundary data depends on the geometry, e.g., the node coordinates.
    reinitialize_boundaries!(semi.boundary_conditions, cache)

    return nothing
end

function deform_mesh!(mesh::T8codeMesh, equations, dg::DGSEM, cache, brep_file,
                      deformation_callback)
    @unpack rbf_type, scale_factor_support_radius = deformation_callback

    # Deform the coarse mesh, which updates its vertices and its CAD geometry.
    # The cmesh is replicated on all ranks, so all ranks deform it in the same way.
    cmesh = t8_forest_get_cmesh(mesh.forest)
    @trixi_timeit timer() "t8code" begin
        t8_cmesh_deform_to_brep(cmesh, brep_file, rbf_type, scale_factor_support_radius)
    end

    # Recompute the geometry of the trees. In t8code, the reference space is [0, 1].
    @trixi_timeit timer() "tree node coordinates" begin
        nodes = 0.5f0 .* (mesh.nodes .+ 1)
        calc_tree_node_coordinates!(mesh.tree_node_coordinates, cmesh, nodes)
    end

    # The number of elements and their connectivity do not change, but all geometric
    # data (node coordinates, metric terms, normal directions, ...) has to be updated.
    @trixi_timeit timer() "reinitialize containers" begin
        reinitialize_containers!(mesh, equations, dg, cache)
    end

    # Write the deformed mesh with the next solution file.
    mesh.unsaved_changes = true

    return nothing
end
end # @muladd
