# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 2 - Regression 2.2: Goals Scored
# ============================================================
# This small driver runs the shared, date-grouped modelling workflow.

from importlib import import_module


def main():
    model_module = import_module("09_objective2_model_comparison")
    model_module.run("2_2")


if __name__ == "__main__":
    main()
