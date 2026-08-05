#!/usr/bin/env python3
"""Check VIF3D spectral energy-dissipation diagnostics.

For incompressible Navier-Stokes in this convention,
    dE/dt = -2 * nu * Z,
where E is kinetic energy and Z is enstrophy. The diagnostics CSV reports
the instantaneous rate and a trapezoidal time integral of that rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate VIF3D spectral energy-dissipation CSV columns."
    )
    parser.add_argument("csv", type=Path, help="spectral_diagnostics_3d.csv")
    parser.add_argument("--nu", type=float, required=True, help="viscosity")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-10,
        help="relative tolerance for diagnostic consistency checks",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="absolute tolerance for diagnostic consistency checks",
    )
    return parser.parse_args()


def close(a: float, b: float, rtol: float, atol: float) -> bool:
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def main() -> int:
    args = parse_args()
    rows = []
    with args.csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "time",
            "spectral_energy",
            "spectral_enstrophy",
            "spectral_energy_dissipation_rate",
            "spectral_cumulative_energy_dissipation",
            "spectral_energy_loss",
            "spectral_energy_dissipation_balance_error",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            print(f"missing columns: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
        rows = list(reader)

    if not rows:
        print("CSV has no data rows", file=sys.stderr)
        return 2

    initial_energy = float(rows[0]["spectral_energy"])
    previous_time = float(rows[0]["time"])
    previous_rate = 2.0 * args.nu * float(rows[0]["spectral_enstrophy"])
    cumulative = 0.0

    for index, row in enumerate(rows):
        time = float(row["time"])
        energy = float(row["spectral_energy"])
        enstrophy = float(row["spectral_enstrophy"])
        rate = 2.0 * args.nu * enstrophy

        if index > 0:
            dt = time - previous_time
            if dt > 0.0:
                cumulative += 0.5 * (previous_rate + rate) * dt

        reported_rate = float(row["spectral_energy_dissipation_rate"])
        reported_cumulative = float(row["spectral_cumulative_energy_dissipation"])
        reported_loss = float(row["spectral_energy_loss"])
        reported_balance_error = float(row["spectral_energy_dissipation_balance_error"])

        expected_loss = initial_energy - energy
        expected_balance_error = expected_loss - cumulative

        checks = (
            ("rate", reported_rate, rate),
            ("cumulative", reported_cumulative, cumulative),
            ("loss", reported_loss, expected_loss),
            ("balance_error", reported_balance_error, expected_balance_error),
        )
        for name, reported, expected in checks:
            if not close(reported, expected, args.rtol, args.atol):
                print(
                    f"row {index}: {name} mismatch: reported={reported:.17e} "
                    f"expected={expected:.17e}",
                    file=sys.stderr,
                )
                return 1

        previous_time = time
        previous_rate = rate

    print(f"validated {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
