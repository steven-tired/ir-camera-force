#!/usr/bin/env python3
"""Print the completed feasibility report to the terminal."""

from pathlib import Path


def main() -> None:
    report = Path(__file__).with_name("FLIR_RADIOMETRIC_FEASIBILITY_REPORT.md")
    print(report.read_text(), end="")


if __name__ == "__main__":
    main()
