"""Approximate local VLNVerse MDL colors/textures in Blender; never edit assets.

This is a preview adapter, not an MDL interpreter or an Isaac RTX appearance
reproduction. Imported mesh UVs are retained; unsupported material features and
ambiguous short-name matches are reported explicitly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


def uv_mapping_parameters(scale, offset, rotation_rad):
    """Express center-pivot UV rotation as Blender's scale/rotate/translate."""
    cosine, sine = math.cos(rotation_rad), math.sin(rotation_rad)
    x, y = offset[0] - .5, offset[1] - .5
    return {
        "scale": [float(scale[0]), float(scale[1]), 1.],
        "rotation": [0., 0., float(rotation_rad)],
        "location": [.5 + cosine * x - sine * y, .5 + sine * x + cosine * y, 0.],
    }


def _value(prim, name, default=None):
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    return default if value is None else value


def _texture(prim, name):
    value = _value(prim, name)
    if value is None or not getattr(value, "path", ""):
        return None
    # USD resolves the path against its authoring layer, which may be a referenced
    # mesh USD rather than the top-level scene directory.
    resolved = getattr(value, "resolvedPath", "")
    if resolved and Path(resolved).is_file():
        return str(Path(resolved).resolve())
    return None


def _spec(material):
    prim = material.GetPrim()
    shaders = [child for child in prim.GetChildren()
               if _value(child, "info:mdl:sourceAsset:subIdentifier")]
    if not shaders:
        return None  # Preserve already imported USD Preview Surface materials.
    shader = shaders[0]
    kind = str(_value(shader, "info:mdl:sourceAsset:subIdentifier"))
    notes = []
    spec = dict(usd_path=str(prim.GetPath()), name=str(prim.GetName()), shader=kind,
                base_color=[.65, .65, .65], texture_path=None, roughness=.55, metallic=0.,
                transmission=0., uv_mapping=uv_mapping_parameters((1, 1), (0, 0), 0),
                diffuse_supported=True, limitations=notes)
    if kind.startswith(("M_BaseMaterial", "M_TranslucentMaterial")):
        if "_v2" in kind:
            color = _value(shader, "inputs:ColorColor", (.65, .65, .65, 1))
            mode = float(_value(shader, "inputs:ColorIsTexType", 0))
            texture_input = "inputs:ColorTex"
            uvw = _value(shader, "inputs:ColorUVW", (1, 1, 0, 1))
            offset = _value(shader, "inputs:ColorOffset", (0, 0, 0, 1))
            # The generated MDL flips V before and after its transform; the two
            # flips cancel but reverse the rotation direction.
            spec["uv_mapping"] = uv_mapping_parameters(uvw, offset, -2 * math.pi * uvw[2])
            spec["roughness"] = float(_value(shader, "inputs:RoughnessFloat", .55))
            spec["metallic"] = float(_value(shader, "inputs:MetallicFloat", 0))
            if mode not in (0., 1.):
                spec["diffuse_supported"] = False
                notes.append("view_dependent_color_falloff_replaced_by_primary_color")
        else:
            color = _value(shader, "inputs:BaseColor_Color", (.65, .65, .65, 1))
            mode = float(_value(shader, "inputs:IsBaseColorTex", 0))
            texture_input = "inputs:BaseColor_Tex"
            uva = _value(shader, "inputs:BaseColor_UVA", (1, 1, 0, 0))
            # The earlier shader uses radians and scales about the flipped-V
            # origin without v2's compensating offset.
            spec["uv_mapping"] = uv_mapping_parameters(uva, (0, 1 - uva[1]), uva[2])
            spec["metallic"] = float(_value(shader, "inputs:Metallic_Color", (0,))[0])
            spec["roughness"] = 1 - float(_value(shader, "inputs:Gloss_Color", (.45,))[0])
        spec["base_color"] = [float(component) for component in color[:3]]
        if mode == 1:
            spec["texture_path"] = _texture(shader, texture_input)
            if spec["texture_path"] is None:
                spec["diffuse_supported"] = False
                notes.append("diffuse_texture_unresolved_or_missing")
        if "Translucent" in kind:
            spec["transmission"] = .9
            notes.append("translucent_mdl_replaced_by_principled_transmission")
        notes.append("normal_emission_fresnel_and_parameter_textures_not_evaluated")
    elif kind in ("OmniPBR", "OmniSurface", "OmniGlass"):
        color_name = "inputs:diffuse_color" if kind == "OmniPBR" else "inputs:diffuse_reflection_color"
        color = _value(shader, color_name, (.8, .8, .8))
        spec["base_color"] = [float(component) for component in color[:3]]
        tex_name = "inputs:diffuse_texture" if kind == "OmniPBR" else "inputs:diffuse_reflection_color_image"
        spec["texture_path"] = _texture(shader, tex_name)
        spec["roughness"] = float(_value(shader, "inputs:roughness_constant",
                                          _value(shader, "inputs:specular_reflection_roughness", .55)))
        if kind == "OmniGlass":
            spec.update(base_color=[.95, .98, 1.], transmission=.98, roughness=.05)
            notes.append("glass_mdl_replaced_by_principled_transmission")
        notes.append("omni_shader_features_approximated")
    else:
        spec["diffuse_supported"] = False
        notes.append("unsupported_mdl_shader_replaced_by_neutral_color")
    for key in ("metallic", "roughness", "transmission"):
        spec[key] = max(0., min(1., spec[key]))
    return spec


def extract_materials(scene_path):
    """Return specs keyed by USD material path and a JSON-serializable audit."""
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(Path(scene_path).resolve()))
    if stage is None:
        raise ValueError(f"Could not open USD scene: {scene_path}")
    specs = {}
    total = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            total += 1
            spec = _spec(UsdShade.Material(prim))
            if spec is not None:
                specs[spec["usd_path"]] = spec
    unsupported = sum(not spec["diffuse_supported"] for spec in specs.values())
    report = dict(source_scene=str(Path(scene_path).resolve()), material_prim_count=total,
                  mdl_material_prim_count=len(specs), unsupported_diffuse_prim_count=unsupported,
                  unsupported_diffuse_prim_fraction=unsupported / max(1, len(specs)),
                  shader_counts=dict(Counter(spec["shader"] for spec in specs.values())))
    return specs, report


def apply_material(material, spec):
    """Install a Principled preview node graph; leave mesh UV layers untouched."""
    import bpy

    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*spec["base_color"], 1.)
    bsdf.inputs["Roughness"].default_value = spec["roughness"]
    bsdf.inputs["Metallic"].default_value = spec["metallic"]
    bsdf.inputs["Transmission Weight"].default_value = spec["transmission"]
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    material.diffuse_color = (*spec["base_color"], 1.)
    material["vlnverse_material_approximation"] = True
    material["vlnverse_usd_material_path"] = spec["usd_path"]
    if spec["texture_path"]:
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = bpy.data.images.load(spec["texture_path"], check_existing=True)
        texture.image.colorspace_settings.name = "sRGB"
        texture.extension = "REPEAT"
        coordinates = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.vector_type = "POINT"
        for name in ("scale", "rotation", "location"):
            mapping.inputs[name.title()].default_value = spec["uv_mapping"][name]
        links.new(coordinates.outputs["UV"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])


def attach_imported_materials(specs):
    """Match imported short names only if all USD candidates agree visually."""
    import bpy

    by_name = defaultdict(list)
    for spec in specs.values():
        by_name[spec["name"]].append(spec)
    applied, ambiguous, unmatched = [], [], []
    unsupported = textured = 0
    for material in list(bpy.data.materials):
        base_name = re.sub(r"\.\d{3,}$", "", material.name)
        matches = by_name.get(material.name, by_name.get(base_name, []))
        if not matches:
            unmatched.append(material.name)
            continue
        variants = {json.dumps({key: value for key, value in spec.items()
                                if key not in ("usd_path", "name")}, sort_keys=True) for spec in matches}
        if len(variants) != 1:
            ambiguous.append(material.name)
            continue
        spec = matches[0]
        apply_material(material, spec)
        applied.append(material.name)
        unsupported += not spec["diffuse_supported"]
        textured += spec["texture_path"] is not None
    return dict(imported_material_count=len(bpy.data.materials), applied_material_count=len(applied),
                textured_applied_material_count=textured, unsupported_diffuse_applied_count=unsupported,
                unsupported_diffuse_applied_fraction=unsupported / max(1, len(applied)),
                ambiguous_imported_names=ambiguous, unmatched_imported_names=unmatched,
                uv_policy="Preserve imported active mesh UVs; apply source diffuse UV transform in nodes.")


def apply_usd_materials(scene_path):
    """Public renderer entry point. Assumes bpy's USD importer already ran."""
    specs, report = extract_materials(scene_path)
    report.update(attach_imported_materials(specs))
    report.update(
        approximation=True, isaac_rtx_equivalent=False,
        note="Diffuse color/texture preview with approximate roughness, metal and glass; no MDL evaluation.",
        unsupported_features=["view-dependent color falloff", "normal maps", "emission textures",
                              "roughness/metallic textures", "original MDL transmission/scattering"])
    return report
