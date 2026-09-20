import taichi as ti
import numpy as np

# -----------------------------------------------------------------------------
# 1. ИНИЦИАЛИЗАЦИЯ ТАИЧИ И ПАРАМЕТРОВ СЕТКИ
# -----------------------------------------------------------------------------
ti.init(arch=ti.gpu)

dim = 3
n_particles = 25000
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4

# Параметры модели Джонсона-Кука для меди (Cu-ETP / М1)
JC_A = 90.0    # Начальный предел текучести (МПа)
JC_B = 292.0   # Модуль деформационного упрочнения (МПа)
JC_n = 0.31    # Показатель степени упрочнения
p_rho = 8.96   # Плотность меди (г/см3)

p_vol = (dx * 0.5) ** 3
p_mass = p_vol * p_rho
E, nu = 110e3, 0.34  # Модуль Юнга (МПа) и коэффициент Пуассона
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# -----------------------------------------------------------------------------
# 2. ПОЛЯ И ПЕРЕМЕННЫЕ
# -----------------------------------------------------------------------------
x = ti.Vector.field(dim, float, shape=n_particles)      # Позиции частиц
v = ti.Vector.field(dim, float, shape=n_particles)      # Скорости частиц
C = ti.Matrix.field(dim, dim, float, shape=n_particles)  # Матрица деформации
F = ti.Matrix.field(dim, dim, float, shape=n_particles)  # Градиент деформации
equivalent_plastic_strain = ti.field(float, shape=n_particles) # Накопленная деформация

grid_v = ti.Vector.field(dim, float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(float, shape=(n_grid, n_grid, n_grid))

# Параметры резания
tool_pos = ti.Vector.field(dim, float, shape=())
tool_speed = ti.field(float, shape=())    # Скорость резания V (X)
cut_depth = ti.field(float, shape=())     # Глубина t (Z)
feed_rate = ti.field(float, shape=())     # Подача S (Y)

# -----------------------------------------------------------------------------
# 3. ИНИЦИАЛИЗАЦИЯ И СБРОС
# -----------------------------------------------------------------------------
@ti.kernel
def reset_simulation():
    tool_pos[None] = [0.15, 0.2, 0.5 + cut_depth[None]]
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.5 + 0.25,  # Длина (X)
            ti.random() * 0.4 + 0.3,   # Ширина / Подача (Y)
            ti.random() * 0.25 + 0.35  # Высота (Z)
        ]
        v[i] = [0, 0, 0]
        F[i] = ti.Matrix.identity(float, dim)
        C[i] = ti.Matrix.zero(float, dim, dim)
        equivalent_plastic_strain[i] = 0.0

# -----------------------------------------------------------------------------
# 4. ФИЗИЧЕСКИЙ РЕШАТЕЛЬ (3D MPM + Джонсон-Кук)
# -----------------------------------------------------------------------------
@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # Движение резца
    tool_pos[None] += ti.Vector([tool_speed[None], feed_rate[None], 0.0]) * dt

    # P2G: Перенос с частиц на 3D-сетку
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        
        # Пластическое упрочнение меди (Джонсон-Кук)
        eps_p = equivalent_plastic_strain[p]
        yield_stress = JC_A + JC_B * (eps_p ** JC_n)
        
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            J *= sig[d, d]

        stress = 2 * mu_0 * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * lambda_0 * (J - 1) * J
        equivalent_plastic_strain[p] += dt * 0.5 * (C[p].norm() + 1e-5)

        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # Взаимодействие заготовки с основанием и 3D-резцом
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            pos = ti.Vector([i, j, k]) * dx
            
            # Закрепление дна
            if pos.z < 0.36:
                grid_v[i, j, k] = [0, 0, 0]

            # Границы 3D-клина резца
            t_p = tool_pos[None]
            if pos.x > t_p.x - (pos.z - t_p.z) * 0.7 and pos.x < t_p.x + 0.25 and pos.z > t_p.z and pos.y > t_p.y - 0.25 and pos.y < t_p.y + 0.25:
                grid_v[i, j, k] = ti.Vector([tool_speed[None], feed_rate[None], 0.0])

    # G2P: Перенос с 3D-сетки обратно на частицы
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

# -----------------------------------------------------------------------------
# 5. ИНТЕРФЕЙС И 3D-ОТРИСОВКА (GUI)
# -----------------------------------------------------------------------------
window = ti.ui.Window("CuCut-CAE: 3D Copper Cutting Simulator", (1024, 768))
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.Camera()

# Начальные значения
tool_speed[None] = 0.25
feed_rate[None] = 0.01
cut_depth[None] = -0.05

reset_simulation()

while window.running:
    # Шаг симуляции
    for _ in range(10):
        substep()

    # Позиционирование 3D-камеры
    camera.position(1.2, 1.2, 1.2)
    camera.lookat(0.4, 0.4, 0.4)
    scene.set_camera(camera)

    # Освещение
    scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
    scene.ambient_light((0.3, 0.3, 0.3))

    # Визуализация частиц меди (исправленный метод particles)
    scene.particles(x, radius=0.006, color=(0.72, 0.45, 0.2))

    canvas.scene(scene)

    # --- ИНТЕРФЕЙС УПРАВЛЕНИЯ (GUI) НА АНГЛИЙСКОМ ---
    window.GUI.begin("Cutting Parameters", 0.02, 0.02, 0.32, 0.28)
    window.GUI.text("CuCut-CAE v1.0")
    tool_speed[None] = window.GUI.slider_float("Cutting Speed V (X)", tool_speed[None], 0.05, 0.8)
    feed_rate[None] = window.GUI.slider_float("Feed Rate S (Y)", feed_rate[None], 0.0, 0.05)
    cut_depth[None] = window.GUI.slider_float("Cut Depth t (Z)", cut_depth[None], -0.1, 0.0)
    
    if window.GUI.button("Reset Simulation"):
        reset_simulation()
    window.GUI.end()

    window.show()