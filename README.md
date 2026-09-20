# CuCut-CAE: 3D Micro-Cutting Simulator for Copper

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen?style=for-the-badge&logo=github)](https://timur830907.github.io/CuCut-CAE/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Taichi](https://img.shields.io/badge/Physics-Taichi%20Lang-orange?style=for-the-badge)](https://www.taichi-lang.org/)

**CuCut-CAE** is an open-source 3D Computer-Aided Engineering (CAE) simulator designed for modeling micro-machining, chip formation, and surface structure evolution during copper cutting.

> 🚀 **Try Interactive Web App:** [https://timur830907.github.io/CuCut-CAE/](https://timur830907.github.io/CuCut-CAE/)

---

## Features
- **3D Material Point Method (MPM):** Handles large plastic deformations and chip separation without mesh distortion.
- **Johnson-Cook Constitutive Model:** Accurate plastic flow parameters for copper (Cu-ETP/M1).
- **GPU Acceleration:** Powered by [Taichi Lang](https://www.taichi-lang.org/) via Vulkan/CUDA backends.
- **Force Analytics & Heatmap:** Real-time calculation of $F_x$, $F_z$ cutting forces and plastic strain visualization.
- **VTK Export:** Export simulation frames for post-processing in **ParaView**.

---

## Quick Start (Desktop Python GUI)

### Prerequisites
- Python 3.8+
- GPU with Vulkan or CUDA support

### Installation
```bash
git clone [https://github.com/timur830907/CuCut-CAE.git](https://github.com/timur830907/CuCut-CAE.git)
cd CuCut-CAE
pip install -r requirements.txt