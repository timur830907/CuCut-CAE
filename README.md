# CuCut-CAE: 3D Micro-Cutting Simulator for Copper

**CuCut-CAE** is an open-source 3D Computer-Aided Engineering (CAE) simulator designed for modeling micro-machining, chip formation, and surface structure evolution during copper cutting.

## Features
- **3D Material Point Method (MPM):** Handles large plastic deformations and chip separation without mesh distortion.
- **Johnson-Cook Constitutive Model:** Accurate constitutive plastic flow parameters for copper (Cu-ETP/M1).
- **GPU Acceleration:** Powered by [Taichi Lang](https://www.taichi-lang.org/) via Vulkan/CUDA backends.
- **Interactive Parameters:** Real-time adjustment of cutting speed ($V$), feed rate ($S$), and depth of cut ($t$).

## Quick Start

### Prerequisites
- Python 3.8+
- GPU with Vulkan or CUDA support

### Installation
```bash
pip install taichi numpy matplotlib