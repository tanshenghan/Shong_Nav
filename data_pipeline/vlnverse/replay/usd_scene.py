"""Import visible static USD meshes with explicit composed world transforms.

Blender 4.2's USD importer lost ancestor transforms on the local VLNVerse assets.
This preview adapter reads the composed USD stage directly, preserves indexed
mesh UVs, and binds approximate materials by full USD path. It never edits USD.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from data_pipeline.vlnverse.replay.usd_materials import apply_material, extract_materials


def loop_values(values, interpolation, point_indices, face_counts, point_count):
    """Expand a flattened USD primvar to Blender's face-corner storage."""
    values = np.asarray(values)
    if interpolation == "faceVarying" and len(values) == len(point_indices):
        return values
    if interpolation in ("vertex", "varying") and len(values) == point_count:
        return values[point_indices]
    if interpolation == "uniform" and len(values) == len(face_counts):
        return np.repeat(values, face_counts, axis=0)
    if interpolation == "constant" and len(values) == 1:
        return np.repeat(values, len(point_indices), axis=0)
    raise ValueError(f"unsupported or inconsistent primvar interpolation: {interpolation}")


def import_static_scene(scene_path):
    """Populate the current bpy scene; return bounds/material/unsupported audit."""
    import bpy
    from mathutils import Matrix, Vector
    from pxr import Usd, UsdGeom, UsdShade

    path = Path(scene_path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD scene: {path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("static VLNVerse preview requires an explicit Z-up stage")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.) > 1e-9:
        raise ValueError("static VLNVerse preview requires a metre-based stage")
    specs, material_audit = extract_materials(path)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    material_cache, applied_specs = {}, {}
    objects, expected_bounds = [], []
    skipped, uv_interpolations = Counter(), Counter()
    source_points = source_faces = uv_layers = custom_normals = 0
    unsupported = Counter()

    def get_material(material):
        key = str(material.GetPath()) if material else "__unbound_neutral__"
        if key not in material_cache:
            spec = specs.get(key)
            if spec is None:
                spec = dict(usd_path=key, name=key.rsplit("/", 1)[-1], shader="unsupported_or_unbound",
                            base_color=[.65, .65, .65], texture_path=None, roughness=.55,
                            metallic=0., transmission=0., diffuse_supported=False)
            converted = bpy.data.materials.new(spec["name"])
            apply_material(converted, spec)
            material_cache[key], applied_specs[key] = converted, spec
        return material_cache[key]

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying():
            raise ValueError(f"animated transforms are unsupported: {prim.GetPath()}")
        if not prim.IsA(UsdGeom.Mesh):
            kind = str(prim.GetTypeName())
            if kind.endswith("Light") or kind in ("Camera", "Volume", "BasisCurves", "Points"):
                skipped[kind] += 1
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            skipped["invisible_mesh"] += 1
            continue
        if imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            skipped["nonrender_purpose_mesh"] += 1
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().ValueMightBeTimeVarying():
            raise ValueError(f"animated points are unsupported: {prim.GetPath()}")
        if prim.GetRelationship("skel:skeleton") or prim.HasAttribute("skel:blendShapes"):
            raise ValueError(f"skeletal/blendshape geometry is unsupported: {prim.GetPath()}")
        points_value = mesh.GetPointsAttr().Get()
        counts_value, indices_value = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if points_value is None or counts_value is None or indices_value is None:
            raise ValueError(f"mesh lacks authored points/topology: {prim.GetPath()}")
        points = np.asarray(points_value, dtype=np.float32)
        counts = np.asarray(counts_value, dtype=np.int32)
        indices = np.asarray(indices_value, dtype=np.int32)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(points).all() or not len(counts) or (counts < 3).any()
                or counts.sum() != len(indices) or (indices < 0).any() or (indices >= len(points)).any()):
            raise ValueError(f"invalid mesh points/topology: {prim.GetPath()}")
        if mesh.GetHoleIndicesAttr().Get():
            raise ValueError(f"hole-index topology is unsupported: {prim.GetPath()}")
        if mesh.GetSubdivisionSchemeAttr().Get() not in (None, UsdGeom.Tokens.none):
            unsupported["subdivision_rendered_as_control_mesh"] += 1
        starts = np.r_[0, np.cumsum(counts[:-1])].astype(np.int32)
        order = np.arange(len(indices))
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            order = np.concatenate([order[start:start + count][::-1]
                                    for start, count in zip(starts, counts)])
        converted = bpy.data.meshes.new(str(prim.GetName()))
        converted.vertices.add(len(points))
        converted.vertices.foreach_set("co", points.ravel())
        converted.loops.add(len(indices))
        converted.loops.foreach_set("vertex_index", indices[order])
        converted.polygons.add(len(counts))
        converted.polygons.foreach_set("loop_start", starts)
        converted.polygons.foreach_set("loop_total", counts)
        converted.update(calc_edges=True)
        for var in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if str(var.GetTypeName()) not in ("texCoord2f[]", "texCoord2d[]", "float2[]", "double2[]"):
                continue
            values = var.ComputeFlattened()
            if values is None:
                raise ValueError(f"UV primvar has no values: {var.GetAttr().GetPath()}")
            expanded = loop_values(np.asarray(values, dtype=np.float32), str(var.GetInterpolation()),
                                   indices, counts, len(points))
            if expanded.shape != (len(indices), 2) or not np.isfinite(expanded).all():
                raise ValueError(f"invalid UV coordinates: {var.GetAttr().GetPath()}")
            uv = converted.uv_layers.new(name=str(var.GetPrimvarName()))
            uv.data.foreach_set("uv", np.ascontiguousarray(expanded[order]).ravel())
            uv_layers += 1
            uv_interpolations[str(var.GetInterpolation())] += 1
        if converted.uv_layers.get("st"):
            converted.uv_layers.active = converted.uv_layers["st"]
            converted.uv_layers["st"].active_render = True
        normals = mesh.GetNormalsAttr().Get()
        if normals is not None and len(normals):
            expanded = loop_values(np.asarray(normals, dtype=np.float32), str(mesh.GetNormalsInterpolation()),
                                   indices, counts, len(points))
            if expanded.shape != (len(indices), 3) or not np.isfinite(expanded).all():
                raise ValueError(f"invalid normals: {prim.GetPath()}")
            converted.normals_split_custom_set(expanded[order].tolist())
            custom_normals += 1
        else:
            unsupported["authored_normals_missing_blender_generated"] += 1
        material_api = UsdShade.MaterialBindingAPI(prim)
        bound = material_api.ComputeBoundMaterial()[0]
        converted.materials.append(get_material(bound))
        for subset in material_api.GetMaterialBindSubsets():
            bound = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            converted.materials.append(get_material(bound))
            slot = len(converted.materials) - 1
            for face in subset.GetIndicesAttr().Get():
                if not 0 <= face < len(converted.polygons):
                    raise ValueError(f"material subset index outside mesh: {prim.GetPath()}")
                converted.polygons[face].material_index = slot
        obj = bpy.data.objects.new(str(prim.GetName()), converted)
        bpy.context.scene.collection.objects.link(obj)
        # USD matrices use row vectors; Blender uses column vectors.
        obj.matrix_world = Matrix(np.asarray(transforms.GetLocalToWorldTransform(prim)).T.tolist())
        obj["usd_prim_path"] = str(prim.GetPath())
        objects.append(obj)
        bounds = boxes.ComputeWorldBound(prim).ComputeAlignedRange()
        expected_bounds.extend((list(bounds.GetMin()), list(bounds.GetMax())))
        source_points += len(points)
        source_faces += len(counts)
    if not objects:
        raise ValueError("USD scene contains no supported visible mesh geometry")
    bpy.context.view_layer.update()
    actual = np.array([list(obj.matrix_world @ Vector(corner))
                       for obj in objects for corner in obj.bound_box])
    expected = np.asarray(expected_bounds)
    actual_bounds, expected_bounds = [actual.min(axis=0), actual.max(axis=0)], [expected.min(axis=0), expected.max(axis=0)]
    error = float(np.max(np.abs(np.asarray(actual_bounds) - expected_bounds)))
    if error > 5e-4:
        raise ValueError(f"USD/Blender world bounds disagree by {error} metres")
    unsupported_materials = sum(not spec["diffuse_supported"] for spec in applied_specs.values())
    material_audit.update(
        approximation=True, isaac_rtx_equivalent=False, binding_policy="full USD material path",
        applied_material_count=len(material_cache),
        textured_applied_material_count=sum(bool(spec["texture_path"]) for spec in applied_specs.values()),
        unsupported_diffuse_applied_count=unsupported_materials,
        unsupported_diffuse_applied_fraction=unsupported_materials / max(1, len(material_cache)),
        ambiguous_imported_names=[], unmatched_imported_names=[])
    return dict(importer="composed_usd_static_meshes_explicit_world_matrices",
                scene_usd=str(path), meters_per_unit=1., up_axis="Z", mesh_objects=len(objects),
                vertex_count=source_points, polygon_count=source_faces, uv_layers=uv_layers,
                uv_interpolations=dict(uv_interpolations), meshes_with_authored_normals=custom_normals,
                usd_world_bounds=np.asarray(expected_bounds).tolist(),
                blender_world_bounds=np.asarray(actual_bounds).tolist(), bounds_max_error_m=error,
                ignored_scene_primitives=dict(skipped), unsupported_features=dict(unsupported),
                materials=material_audit, physics_replayed=False, source_assets_modified=False)
