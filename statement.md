# Problem Statement & Scope

## Problem Statement
Traditional manual attendance logging is time-consuming and prone to manual entry errors. Existing high-end automated facial recognition systems often rely on heavy deep-learning frameworks that require extensive compute resources and third-party C++ bindings. There is a need for a lightweight, dependency-minimal image feature recognition system that runs efficiently via CLI and logs attendance dynamically.

## Scope of the Project
The Smart Attendance System focuses on:
- Batch profile feature extraction using low-overhead Sobel gradient filters.
- Fast matrix matching using Euclidean distance metrics.
- Automated dynamic logging into CSV format with duplicate entry prevention.
- Command-line interface (CLI) driven execution suitable for headless environments.

## Target Users
- Academic institutions for lab session check-ins.
- Event organizers tracking participant registration.
- System administrators running headless or resource-constrained monitoring tools.

## High-Level Features
1. **Profile Directory Parsing:** Automated parsing and normalization of stored target faces.
2. **Gradient Feature Extraction:** OpenCV-driven Sobel edge magnitude processing.
3. **Similarity Engine:** Vectorized Euclidean distance calculation.
4. **Attendance Logging & Validation:** Dynamic CSV output with timestamping and session duplicate skipping.