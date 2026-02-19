from __future__ import annotations
import types, typing as T
import pprint
from wplib import log





def apply_skinned_meshes(from_meshes:list[tuple[str]], to_mesh:tuple[str]):
    """this isn't clean yet, passing in (skc, mesh) tuples etc -
    just a first sketch for what can be done with this system
    """
    print("applying skinned meshes to: ", to_mesh)
    start_t = datetime.now()

    base_mesh_fn = get_mesh_fn(to_mesh[1])
    base_point_arr = mesh_point_arr(base_mesh_fn)
    base_kd_tree = spatial.KDTree(base_point_arr)
    base_skin_fn = oma.MFnSkinCluster(lib.get_MObject(to_mesh[0]))
    base_mesh_shape_dag = lib.get_MDagPath(lib.get_shape(to_mesh[1]))

    # get global influence map to convert indices between skins
    all_influences, skc_inf_map = get_all_scene_skin_drivers()

    # check that ALL influences are active on body mesh
    #base_infs = set(cmds.skinCluster(to_mesh[0], q=1, inf=1) or [])
    base_infs = set(get_skin_driver_names(to_mesh[0]))

    # add any missing to the body mesh
    for i in all_influences:
        if i in base_infs:
            continue
        size = cmds.getAttr(to_mesh[0] + ".bindPreMatrix", size=1) # bindPreMatrix has priority on how many actual influences there are
        lib.set_world_mat(to_mesh[0] + ".bindPreMatrix[{}]".format(size), lib.get_world_mat(i).inverse())
        #cmds.connectAttr(i.split(".")[0] + ".worldInverseMatrix[0]", to_mesh[0] + ".bindPreMatrix[{}]".format(size))
        cmds.connectAttr(i, to_mesh[0] + ".matrix[{}]".format(size))
        #lib.set_world_mat(to_mesh[0] + ".bindPreMatrix[{}]".format(size), lib.get_world_mat(i).inverse())

        base_infs.add(i)
    connect_t = datetime.now()
    print("connected all influences in", connect_t - start_t)
    #return
    base_infs = get_skin_driver_names(to_mesh[0]) # all infs now connected
    base_skin = ArraySkinDense(to_mesh[0]) # with many smaller meshes it actually seems more efficient to use a dense array here
    #base_skin = ArraySkinSparse(to_mesh[0])

    print("begin weight composite")

    all_other_point_arr = []
    mesh_indices_map = {}
    start_index = 0
    for (i, (skc, mesh)) in enumerate(from_meshes):
        other_mesh_fn = get_mesh_fn(mesh)
        other_point_arr = mesh_point_arr(other_mesh_fn)
        all_other_point_arr.extend(other_point_arr)
        indices = np.arange(start_index, start_index + other_mesh_fn.numVertices, dtype=int)
        mesh_indices_map[i] = indices
        start_index += other_mesh_fn.numVertices
    collate_pt_t = datetime.now()
    print("collated points in ", collate_pt_t - connect_t, "  querying kd tree")
    all_sampled_indices = base_kd_tree.query(np.array(all_other_point_arr),
                                             k=1)[1]
    query_t = datetime.now()
    print("queried kd tree in  ", query_t - collate_pt_t, "  applying")

    for(i, (skc, mesh)) in enumerate(from_meshes):
        #print("apply mesh", i, skc, mesh)
        if not (i % 20):
            print("apply mesh", i, skc, mesh)
        #print("i, skc, mesh", i, skc, mesh)
        other_skin = ArraySkinDense(skc)
        #other_skin = ArraySkinSparse(skc)

        closest_points_on_to_mesh = all_sampled_indices[ mesh_indices_map[i] ]
        from_skc_index_to_target_index_arr = np.array([base_infs.index(inf) for inf in skc_inf_map[skc]], dtype=int)

        base_skin.weight_arr[closest_points_on_to_mesh, :] = 0.0

        # #### indexing arrays need to have SAME DIMENSION by default - they form SINGLE COORDINATE PATHS, not products

        #print("other weights", other_skin.weight_arr)
        for i, pt in enumerate(closest_points_on_to_mesh):

            base_skin.weight_arr[pt, from_skc_index_to_target_index_arr] = other_skin.weight_arr[i]


    composite_t = datetime.now()
    print("weights composited in", composite_t - query_t,"  applying")
    base_skin.set_weight_arr()

    #set_MFnSkinCluster_array(base_skin_fn, base_skin.weight_arr, base_mesh_shape_dag)

    apply_t = datetime.now()
    print("weights fully applied in", apply_t - composite_t, "total time: ", apply_t - start_t)

    pass
