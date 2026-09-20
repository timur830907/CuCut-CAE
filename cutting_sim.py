import os
import csv
import taichi as ti
import numpy as np

try:
    from pyevtk.hl import pointsToVTK
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

ti.init(arch=ti.gpu)

dim = 3
n_particles = 30000
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4

# -----------------------------------------------------------------------------
# 1. БАЗА МАТЕРИАЛОВ (Johnson-Cook)
# -----------------------------------------------------------------------------
# [A (MPa), B (MPa), n, m, T_melt (C), rho (g/cm3), E (MPa), nu, Cp (J/kgK)]
MATERIALS = {
    "Copper (Cu-ETP)": [90.0, 292.0, 0.31, 1.09, 1083.0, 8.96, 110e3, 0.34, 385.0],
    "Aluminum (6061-T6)": [324.0, 114.0, 0.42, 1.34, 652.0, 2.70, 68.9e3, 0.33, 896.0],
    "Titanium (Ti-6Al-4V)": [1098.0, 1092.0, 0.93, 1.10, 1660.0, 4.43, 113.8e3, 0.34, 526.0],
    "Steel (AISI 1045)": [553.1, 600.8, 0.234, 1.00, 1460.0, 7.85, 200e3, 0.30, 486.0]
}

# Поля состояния
x = ti.Vector.field(dim, float, shape=n_particles)
v = ti.Vector.field(dim, float, shape=n_particles)
C = ti.Matrix.field(dim, dim, float, shape=n_particles)
F = ti.Matrix.field(dim, dim, float, shape=n_particles)
equivalent_plastic_strain = ti.field(float, shape=n_particles)
temperature = ti.field(float, shape=n_particles)
particle_colors = ti.Vector.field(3, float, shape=n_particles)

grid_v = ti.Vector.field(dim, float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(float, shape=(n_grid, n_grid, n_grid))

tool_pos = ti.Vector.field(dim, float, shape=())
tool_speed = ti.field(float, shape=())
cut_depth = ti.field(float, shape=())
feed_rate = ti.field(float, shape=())
rake_angle = ti.field(float, shape=())
edge_radius = ti.field(float, shape=())

Fx_field = ti.field(float, shape=())
Fz_field = ti.field(float, shape=())

# Текущие свойства материала
mat_A = ti.field(float, shape=())
mat_B = ti.field(float, shape=())
mat_n = ti.field(float, shape=())
mat_m = ti.field(float, shape=())
mat_Tmelt = ti.field(float, shape=())
mat_rho = ti.field(float, shape=())
mat_E = ti.field(float, shape=())
mat_nu = ti.field(float, shape=())
mat_Cp = ti.field(float, shape=())

display_mode = 0  # 0: Plastic Strain, 1: Temperature
frame_count = 0
force_history = []

def set_material(name):
    p = MATERIALS[name]
    mat_A[None], mat_B[None], mat_n[None], mat_m[None] = p[0], p[1], p[2], p[3]
    mat_Tmelt[None], mat_rho[None], mat_E[None], mat_nu[None], mat_Cp[None] = p[4], p[5], p[6], p[7], p[8]

@ti.kernel
def reset_simulation():
    tool_pos[None] = [0.15, 0.2, 0.5 + cut_depth[None]]
    Fx_field[None] = 0.0
    Fz_field[None] = 0.0
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.5 + 0.25,
            ti.random() * 0.4 + 0.3,
            ti.random() * 0.25 + 0.35
        ]
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix.identity(float, dim)
        C[i] = ti.Matrix.zero(float, dim, dim)
        equivalent_plastic_strain[i] = 0.0
        temperature[i] = 20.0  # Комнатная температура (C)
        particle_colors[i] = [0.72, 0.45, 0.2]

@ti.kernel
def substep():
    Fx_field[None] = 0.0
    Fz_field[None] = 0.0

    p_vol = (dx * 0.5) ** 3
    p_mass = p_vol * mat_rho[None]
    mu_0 = mat_E[None] / (2 * (1 + mat_nu[None]))
    lambda_0 = mat_E[None] * mat_nu[None] / ((1 + mat_nu[None]) * (1 - 2 * mat_nu[None]))

    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    tool_pos[None] += ti.Vector([tool_speed[None], feed_rate[None], 0.0]) * dt

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        
        # Пластическая деформация
        d_eps = dt * 0.8 * (C[p].norm() + 1e-5)
        equivalent_plastic_strain[p] += d_eps

        # Расчёт выделения тепла (Закон Тейлора-Квинни: 90% работы в тепло)
        d_work = mat_A[None] * d_eps
        dT = (0.90 * d_work) / (mat_rho[None] * mat_Cp[None] * 1e-3)
        temperature[p] = ti.min(mat_Tmelt[None], temperature[p] + dT)

        # Выбор цвета (0: Напряжения, 1: Температура)
        if display_mode == 0:
            val = ti.min(1.0, equivalent_plastic_strain[p] * 0.15)
            particle_colors[p] = [0.72 + val * 0.28, 0.45 * (1.0 - val), 0.2 * (1.0 - val)]
        else:
            t_norm = ti.min(1.0, (temperature[p] - 20.0) / (mat_Tmelt[None] * 0.5 - 20.0))
            particle_colors[p] = [t_norm, 0.2 * (1.0 - t_norm), 1.0 - t_norm]

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            J *= sig[d, d]

        stress = 2 * mu_0 * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * lambda_0 * (J - 1) * J
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    tan_gamma = ti.tan(rake_angle[None] * 3.1415926 / 180.0)

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            pos = ti.Vector([i, j, k]) * dx
            
            if pos.z < 0.36:
                grid_v[i, j, k] = [0, 0, 0]

            t_p = tool_pos[None]
            bound_x = t_p.x - (pos.z - t_p.z) * tan_gamma
            if pos.x > bound_x and pos.x < t_p.x + 0.25 and pos.z > t_p.z and pos.y > t_p.y - 0.25 and pos.y < t_p.y + 0.25:
                delta_v = ti.Vector([tool_speed[None], feed_rate[None], 0.0]) - grid_v[i, j, k]
                Fx_field[None] += grid_m[i, j, k] * ti.abs(delta_v.x) / dt
                Fz_field[None] += grid_m[i, j, k] * ti.abs(delta_v.z) / dt
                grid_v[i, j, k] = ti.Vector([tool_speed[None], feed_rate[None], 0.0])

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

def export_csv():
    os.makedirs("data_output", exist_ok=True)
    with open("data_output/cutting_forces.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Fx_N", "Fz_N"])
        for step, (fx, fz) in enumerate(force_history):
            writer.writerow([step, fx, fz])

# -----------------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ И ИНТЕРФЕЙС
# -----------------------------------------------------------------------------
window = ti.ui.Window("CuCut-CAE v1.3: Multiphysics & Material Suite", (1024, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

set_material("Copper (Cu-ETP)")
tool_speed[None] = 0.25
feed_rate[None] = 0.01
cut_depth[None] = -0.05
rake_angle[None] = 15.0

reset_simulation()

step_count = 0
while window.running:
    for _ in range(10):
        substep()
        step_count += 1
        if step_count % 5 == 0:
            force_history.append((Fx_field[None], Fz_field[None]))

    camera.position(1.2, 1.2, 1.2)
    camera.lookat(0.4, 0.4, 0.4)
    scene.set_camera(camera)
    scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
    scene.ambient_light((0.3, 0.3, 0.3))

    scene.particles(x, radius=0.005, per_vertex_color=particle_colors)
    canvas.scene(scene)

    window.GUI.begin("Multiphysics CAE Suite", 0.02, 0.02, 0.38, 0.48)
    window.GUI.text("CuCut-CAE v1.3 (Thermo-Mechanical)")
    
    tool_speed[None] = window.GUI.slider_float("Cutting Speed V", tool_speed[None], 0.05, 0.8)
    cut_depth[None] = window.GUI.slider_float("Cut Depth t", cut_depth[None], -0.1, 0.0)
    rake_angle[None] = window.GUI.slider_float("Rake Angle Gamma", rake_angle[None], 0.0, 35.0)

    window.GUI.text("-----------------------------")
    window.GUI.text(f"Cutting Force Fx: {Fx_field[None]:.2f} N")
    window.GUI.text(f"Thrust Force Fz:  {Fz_field[None]:.2f} N")

    if window.GUI.button("Export CSV (Forces)"):
        export_csv()

    if window.GUI.button("Reset Simulation"):
        reset_simulation()
        force_history.clear()

    window.GUI.end()
    window.show()