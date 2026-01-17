#Beam M² / BPP Analysis Tool

A fast, cross-platform Python application for beam quality analysis (M², BPP, caustics) based on standard M² measurement files, with optional image-based verification.

Designed for large batch processing and component characterization, the tool prioritizes metadata-first workflows (no image I/O unless explicitly requested), making it suitable for production environments and large datasets.

Key features

Batch and single-file M² analysis

Recursive batch processing with component grouping

Moment-based and image-based width methods (2σ, Gaussian 1/e²)

Interactive caustic plots, histograms, and statistical summaries

Modern, responsive PyQt GUI (PyQt5 / PyQt6 compatible)

Lazy image loading with false-color visualization (Viridis, Inferno, etc.)

Export to Excel and CSV for reporting and trending

Cross-platform: Linux and Windows

Intended use

This tool is intended for engineering, R&D, and production characterization of laser beam quality. It is provided as-is and is not a certified metrology instrument.
