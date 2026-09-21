"""
builder/xml_generator.py  -- v1.0.2
======================================
Sinh file MJCF XML cho Single Twistable Tendon-Driven Continuum Robot.

v1.0.2: Cap nhat:
  [TASK 2] Them STL mesh layer (visual-only):
    - notch.stl cho cac segment giua
    - distal.stl cho segment cuoi (tip)
    - Offset/rotation theo render.m cua bai bao
    - Capsule collision geom giu nguyen (tang hinh, rgba=0 0 0 0)
  [TASK 4] Xay dung lai moi truong 3D theo reference Rigid Body Fundamentals:
    - Floor on dinh tai vi tri co dinh
    - Anh sang chat luong cao (shadowsize=4096)
    - Lighting scheme tu 09_Light/model.xml

Cau truc XML:
  <asset>
    <mesh name="notch_mesh" file="meshes/notch.stl" scale="0.001 ..."/>
    <mesh name="distal_mesh" file="meshes/distal.stl" scale="0.001 ..."/>
    ... materials, textures
  </asset>
  <worldbody>
    <geom type="plane" .../> (floor, stable)
    <light .../> (sun + fills)
    <body name="robot_base">
      ... kinematic chain with dual geoms:
        <geom type="mesh" .../> (visual, group=1)
        <geom type="capsule" rgba="0 0 0 0" .../> (collision, invisible)
      ...
    </body>
  </worldbody>

STL coordinate transform (theo render.m):
  render.m: fv.vertices += [-1.75, -50, -1.75]  -- offset STL toi goc
  render.m: rotate(seg, [1 0 0], -90, ...)       -- quay X -90 do
  render.m: rotate(seg, [0 0 1], +90, ...)       -- quay Z +90 do
  => MuJoCo euler="90 0 -90" (XYZ Euler) cho geom mesh
     pos offset = (0, 0, 0) vi joint frame tu dong tinh
"""

import os
import math
import yaml
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------------------------------------------------------
ROBOT_CONFIGS = {
    1: {"L": 0.035, "m": 11, "OD": 0.0035, "ID": 0.0020},
    2: {"L": 0.050, "m": 16, "OD": 0.0035, "ID": 0.0020},
    3: {"L": 0.055, "m": 24, "OD": 0.0020, "ID": 0.00075},
    4: {"L": 0.098, "m": 32, "OD": 0.0035, "ID": 0.0020},
}

NITI_DENSITY = 6450.0
PI = math.pi


def _load_params(config_path: str) -> dict:
    defaults = {
        "timestep": 0.001,
        "integrator": "implicitfast",
        "bend_stiffness": 0.005,
        "twist_stiffness": 0.050,
        "bend_damping": 0.0005,
        "twist_damping": 0.0010,
        "bend_friction": 0.0002,
        "twist_friction": 0.0005,
        "armature": 1e-6,
        "mass_factor": 0.60,
        "camera_width": 640,
        "camera_height": 360,
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        defaults.update(loaded)
    except FileNotFoundError:
        print(f"[XMLGenerator] Config not found: {config_path}. Using defaults.")
    return defaults


def generate_continuum_robot_xml(
    robot_id: int = 2,
    config_path: str = "configs/robot_params.yaml",
    output_dir: str = "assets",
    use_stl: bool = True,
) -> tuple[str, int]:
    """
    Sinh file MJCF XML cho robot lien tuc PPT v1.0.2.

    v1.0.2 moi:
      - Dual-geom: mesh (visual) + capsule (collision, invisible)
      - Moi truong 3D on dinh: floor co dinh, lighting 4-point
      - STL mesh tu notch.stl / distal.stl voi offset tu render.m

    Tra ve:
      (xml_path, m_notches)
    """
    if robot_id not in ROBOT_CONFIGS:
        raise ValueError(f"robot_id must be 1-4, got: {robot_id}")

    p   = _load_params(config_path)
    cfg = ROBOT_CONFIGS[robot_id]

    L_total   = cfg["L"]
    m_notches = cfg["m"]
    r_o       = cfg["OD"] / 2.0
    r_i       = cfg["ID"] / 2.0
    seg_len   = L_total / m_notches
    tendon_d  = (r_o + r_i) / 2.0

    vol_full  = PI * (r_o**2 - r_i**2) * seg_len
    seg_mass  = max(NITI_DENSITY * vol_full * p["mass_factor"], 5e-5)

    # =========================================================================
    mujoco_el = ET.Element("mujoco", model=f"continuum_robot_v{robot_id}")

    # -- 1. Compiler
    ET.SubElement(mujoco_el, "compiler", angle="degree")

    # -- 2. Option
    ET.SubElement(
        mujoco_el, "option",
        timestep=str(p["timestep"]),
        gravity="0 0 -9.81",
        integrator=p["integrator"],
    )

    # -- 3. Visual quality (theo 09_Light reference)
    visual_el = ET.SubElement(mujoco_el, "visual")
    ET.SubElement(visual_el, "quality", shadowsize="4096")
    ET.SubElement(
        visual_el, "headlight",
        diffuse="0.5 0.5 0.5",
        ambient="0.3 0.3 0.3",
        specular="0 0 0",
    )

    # -- 4. Defaults
    default = ET.SubElement(mujoco_el, "default")

    # Geom mac dinh: capsule cho collision (transparent, invisible)
    # Visual mesh duoc khai bao rieng tung geom
    ET.SubElement(
        default, "geom",
        type="capsule",
        contype="0",
        conaffinity="0",
        rgba="0 0 0 0",           # invisible collision primitive
    )

    bend_cls = ET.SubElement(default, "default", attrib={"class": "bend_joint"})
    ET.SubElement(
        bend_cls, "joint",
        type="hinge",
        stiffness=str(p["bend_stiffness"]),
        damping=str(p["bend_damping"]),
        frictionloss=str(p["bend_friction"]),
        armature=str(p["armature"]),
        limited="true",
        range="-45 45",
    )

    twist_cls = ET.SubElement(default, "default", attrib={"class": "twist_joint"})
    ET.SubElement(
        twist_cls, "joint",
        type="hinge",
        stiffness=str(p["twist_stiffness"]),
        damping=str(p["twist_damping"]),
        frictionloss=str(p["twist_friction"]),
        armature=str(p["armature"]),
        limited="true",
        range="-180 180",
    )

    # -- 5. Assets
    asset = ET.SubElement(mujoco_el, "asset")

    # STL meshes (unit mm in STL file -> scale to m for MuJoCo)
    # Scale: 0.001 chuyen mm -> m
    if use_stl:
        ET.SubElement(asset, "mesh",
                      name="notch_mesh",
                      file="meshes/notch.stl",
                      scale="0.001 0.001 0.001")
        ET.SubElement(asset, "mesh",
                      name="distal_mesh",
                      file="meshes/distal.stl",
                      scale="0.001 0.001 0.001")

    # Checker texture san (theo 09_Light reference + v1.0.1)
    ET.SubElement(asset, "texture",
                  name="checker",
                  type="2d",
                  builtin="checker",
                  width="512", height="512",
                  rgb1="0.20 0.23 0.27",
                  rgb2="0.14 0.17 0.20")
    ET.SubElement(asset, "material",
                  name="floor_mat",
                  texture="checker",
                  texrepeat="8 8",
                  reflectance="0.25",
                  shininess="0.1")

    # Vat lieu NiTi robot - mau xanh bong nhu anh tham chieu
    # Mau phai giong voi render.m: FaceColor [0.6902 0.8784 0.9020]
    ET.SubElement(asset, "material",
                  name="niti_mat",
                  rgba="0.690 0.878 0.902 1.0",
                  shininess="0.85",
                  specular="0.6",
                  reflectance="0.1")

    ET.SubElement(asset, "material",
                  name="tip_mat",
                  rgba="1.0 0.50 0.10 1.0",
                  shininess="0.90",
                  specular="0.7")

    ET.SubElement(asset, "material",
                  name="base_mat",
                  rgba="0.60 0.62 0.65 1.0",
                  shininess="0.6",
                  specular="0.5")

    # -- 6. Worldbody
    worldbody = ET.SubElement(mujoco_el, "worldbody")

    # ── SAN NHA ON DINH (TASK 4) ──────────────────────────────────────────
    # Vi tri co dinh tai z=-0.005 (5mm duoi goc toa do robot)
    # Khong bao gio "bien mat" vi khong phu thuoc vao robot dynamics
    ET.SubElement(
        worldbody, "geom",
        name="floor",
        type="plane",
        size="0.8 0.8 0.1",
        pos="0 0 -0.005",
        material="floor_mat",
        contype="1",
        conaffinity="1",
    )

    # ── LIGHTING SCHEME (TASK 4, theo 09_Light reference) ─────────────────
    # Ambient: giup khong qua toi
    ET.SubElement(
        worldbody, "light",
        name="ambient",
        directional="false",
        pos="0 0 0.5",
        diffuse="0.25 0.25 0.25",
        specular="0 0 0",
        castshadow="false",
    )
    # Sun: directional + castshadow=true (chu dang bo bong)
    ET.SubElement(
        worldbody, "light",
        name="sun",
        directional="true",
        pos="0.3 -0.5 1.0",
        dir="-0.3 0.5 -1.0",
        castshadow="true",
        diffuse="0.85 0.85 0.85",
        specular="0.30 0.30 0.30",
    )
    # Fill tu phia trai (giam vung toi)
    ET.SubElement(
        worldbody, "light",
        name="fill_left",
        directional="true",
        pos="-0.6 0.3 0.5",
        dir="0.6 -0.3 -0.5",
        castshadow="false",
        diffuse="0.28 0.32 0.40",
        specular="0 0 0",
    )
    # Fill tu duoi (tranh vung toi phia duoi robot)
    ET.SubElement(
        worldbody, "light",
        name="fill_bottom",
        directional="true",
        pos="0 0 -0.3",
        dir="0 0 1",
        castshadow="false",
        diffuse="0.10 0.11 0.14",
        specular="0 0 0",
    )

    # ── DE ROBOT (fixed to world) ─────────────────────────────────────────
    base = ET.SubElement(worldbody, "body", name="robot_base", pos="0 0 0")
    # Base geom: cylinder nho (visible)
    ET.SubElement(
        base, "geom",
        name="base_geom",
        type="cylinder",
        size=f"{r_o:.6f} 0.002",
        mass="0.003",
        material="base_mat",
        contype="0",
        conaffinity="0",
        rgba="0.6 0.62 0.65 1.0",
    )
    # Tendon site o +Y offset
    ET.SubElement(
        base, "site",
        name="base_tendon_site",
        pos=f"0 {tendon_d:.6f} 0",
        size="0.0004",
        rgba="1 0.2 0.2 1",
    )
    ET.SubElement(base, "site",
                  name="base_site", pos="0 0 0",
                  size="0.0005", rgba="0.5 0.5 0.5 0.5")

    # ── CHUOI DONG HOC ───────────────────────────────────────────────────
    tendon_sites = ["base_tendon_site"]
    parent       = base

    # STL euler rotation theo render.m:
    #   rotate(seg, [1 0 0], -90, ...)  = X-axis -90 deg
    #   rotate(seg, [0 0 1], +90, ...)  = Z-axis +90 deg
    # Trong MuJoCo angle="degree", euler="X Y Z":
    #   euler="-90 0 90"
    # Nhung can test thuc te de chinh offset Z cho STL
    # STL centroid offset (tu render.m line 4):
    #   fv.vertices += [-1.75, -50, -1.75]  (mm)
    # => Sau khi scale 0.001: centroid offset = [-0.00175, -0.05, -0.00175] m
    # pos trong moi geom = 0 vi joint frame da dich chuyen body

    # Tinh toan offset doc truc Z cho STL
    # STL doc theo Z cua paper = 50mm (height of one notch+segment block)
    # After rotation -90 X, +90 Z, truc Z cua STL thanh truc Y cua MuJoCo
    # Segment height = seg_len; offset de dat STL giua segment
    stl_euler    = "-90 0 90"
    # Fix 1: Offset CAD goc tu render.m line 4:
    #   fv.vertices += [-1.75, -50-base, -1.75]  (mm)
    # Sau khi quay -90 X, +90 Z, vector offset [−1.75, −50, −1.75] mm
    # trong he toa do STL goc chuyen thanh [0.00175, -0.00175, 0.050] m
    # trong he toa do body MuJoCo. (da nhan ma tran quay)
    stl_notch_pos  = "0.00175 -0.00175 0.050"
    stl_distal_pos = "0.00175 -0.00175 0.050"

    for i in range(m_notches):
        is_tip   = (i == m_notches - 1)
        seg_name = f"segment_{i + 1}"

        segment = ET.SubElement(
            parent, "body",
            name=seg_name,
            pos=f"0 0 {seg_len:.6f}",
        )

        # Khop uon (truc X)
        ET.SubElement(
            segment, "joint",
            name=f"bend_x_{i + 1}",
            axis="1 0 0",
            attrib={"class": "bend_joint"},
        )

        # Khop xoan (truc Z)
        ET.SubElement(
            segment, "joint",
            name=f"twist_z_{i + 1}",
            axis="0 0 1",
            attrib={"class": "twist_joint"},
        )

        half_len = seg_len * 0.45

        # ── COLLISION GEOM (capsule, tang hinh rgba=0,0,0,0) ────────────
        # Fix 2: pos=seg_len/2 dat khoi va cham vao chinh giua doan vat ly
        ET.SubElement(
            segment, "geom",
            name=f"col_geom_{i + 1}",
            type="capsule",
            pos=f"0 0 {seg_len/2:.6f}",   # centroid tai giua segment
            size=f"{r_o:.6f} {half_len:.6f}",
            mass=f"{seg_mass:.6e}",
            contype="0",
            conaffinity="0",
            rgba="0 0 0 0",             # hoan toan tang hinh
        )

        # ── VISUAL GEOM (STL mesh hoac capsule fallback) ───────────────
        if use_stl:
            if is_tip:
                ET.SubElement(
                    segment, "geom",
                    name=f"vis_geom_{i + 1}",
                    type="mesh",
                    mesh="distal_mesh",
                    euler=stl_euler,
                    pos=stl_distal_pos,
                    contype="0",
                    conaffinity="0",
                    group="1",
                    material="tip_mat",
                )
            else:
                ET.SubElement(
                    segment, "geom",
                    name=f"vis_geom_{i + 1}",
                    type="mesh",
                    mesh="notch_mesh",
                    euler=stl_euler,
                    pos=stl_notch_pos,
                    contype="0",
                    conaffinity="0",
                    group="1",
                    material="niti_mat",
                )
        else:
            # Fallback: capsule visible (khi khong co STL)
            if is_tip:
                ET.SubElement(
                    segment, "geom",
                    name=f"vis_geom_{i + 1}",
                    type="capsule",
                    size=f"{r_o:.6f} {half_len:.6f}",
                    contype="0", conaffinity="0",
                    group="1",
                    material="tip_mat",
                )
            else:
                ET.SubElement(
                    segment, "geom",
                    name=f"vis_geom_{i + 1}",
                    type="capsule",
                    size=f"{r_o:.6f} {half_len:.6f}",
                    contype="0", conaffinity="0",
                    group="1",
                    material="niti_mat",
                )

        # Tendon site lech tam (+Y) — cho ppt_tendon
        site_name = f"tendon_point_{i + 1}"
        ET.SubElement(
            segment, "site",
            name=site_name,
            pos=f"0 {tendon_d:.6f} 0",
            size="0.00025",
            rgba="1 0.2 0.2 1",
        )
        tendon_sites.append(site_name)

        # Dot cuoi: tip_site + camera
        if is_tip:
            ET.SubElement(
                segment, "site",
                name="tip_site",
                pos=f"0 0 {seg_len / 2:.6f}",
                size="0.001",
                rgba="1 1 0 1",
            )
            ET.SubElement(
                segment, "camera",
                name="endoscope",
                pos=f"0 0 {seg_len:.6f}",
                xyaxes="1 0 0 0 0 1",
                fovy="90",
            )

        parent = segment

    # -- 7. Tendon (ppt_tendon cho uon)
    tendon = ET.SubElement(mujoco_el, "tendon")
    spatial = ET.SubElement(
        tendon, "spatial",
        name="ppt_tendon",
        width="0.00022",
        rgba="1 0.2 0.2 0.9",
    )
    for site in tendon_sites:
        ET.SubElement(spatial, "site", site=site)

    # -- 8. Actuator
    actuator = ET.SubElement(mujoco_el, "actuator")

    # Push-Pull
    ET.SubElement(
        actuator, "motor",
        name="push_pull_motor",
        tendon="ppt_tendon",
        ctrlrange="-5 5",
        gear="1",
    )

    # Twist: mot motor moi segment, tat ca nhan cung ctrl value tu main.py
    for i in range(m_notches):
        ET.SubElement(
            actuator, "motor",
            name=f"twist_motor_{i + 1}",
            joint=f"twist_z_{i + 1}",
            ctrlrange="-0.5 0.5",
            gear="1",
        )

    # -- 9. Sensors
    sensor = ET.SubElement(mujoco_el, "sensor")
    ET.SubElement(sensor, "tendonpos",   name="tendon_length",  tendon="ppt_tendon")
    ET.SubElement(sensor, "tendonvel",   name="tendon_vel",      tendon="ppt_tendon")
    ET.SubElement(sensor, "actuatorfrc", name="push_pull_frc",  actuator="push_pull_motor")
    ET.SubElement(sensor, "actuatorfrc", name="twist_frc",      actuator="twist_motor_1")
    ET.SubElement(sensor, "framepos",    name="tip_pos_sensor",
                  objtype="site", objname="tip_site")
    ET.SubElement(sensor, "framequat",   name="tip_quat_sensor",
                  objtype="site", objname="tip_site")
    ET.SubElement(sensor, "framelinvel", name="tip_linvel_sensor",
                  objtype="site", objname="tip_site")
    ET.SubElement(sensor, "frameangvel", name="tip_angvel_sensor",
                  objtype="site", objname="tip_site")

    # -- 10. Xuat XML
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"robot_model_v{robot_id}.xml")

    raw_xml = ET.tostring(mujoco_el, encoding="unicode")
    pretty  = minidom.parseString(raw_xml).toprettyxml(indent="  ")
    lines   = [ln for ln in pretty.split("\n") if ln.strip()]
    output  = "\n".join(lines)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"[XMLGenerator v1.0.2] Robot {robot_id}: "
          f"L={L_total*1000:.0f}mm, m={m_notches}, "
          f"OD={cfg['OD']*1000:.1f}mm, ID={cfg['ID']*1000:.1f}mm")
    print(f"[XMLGenerator v1.0.2]   seg_len={seg_len*1000:.3f}mm, "
          f"tendon_d={tendon_d*1000:.3f}mm, seg_mass={seg_mass*1000:.3f}g")
    print(f"[XMLGenerator v1.0.2]   [T2] STL mesh: {'ON' if use_stl else 'OFF'} "
          f"(notch.stl + distal.stl)")
    print(f"[XMLGenerator v1.0.2]   [T4] Rebuilt environment: floor + 4-light setup")
    print(f"[XMLGenerator v1.0.2]   Output: {filename}")
    return filename, m_notches


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Robot XML generator v1.0.2")
    parser.add_argument("--id", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-stl", action="store_true")
    args = parser.parse_args()
    use_stl = not args.no_stl
    if args.all:
        for rid in [1, 2, 3, 4]:
            generate_continuum_robot_xml(robot_id=rid, use_stl=use_stl)
    else:
        generate_continuum_robot_xml(robot_id=args.id, use_stl=use_stl)