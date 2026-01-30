"""Contact sequence utilities for MPC configuration."""

from __future__ import annotations

from typing import Any

import numpy as np


def create_contact_sequence(
    total_duration: float, dt: float, phases: dict[str, dict[str, Any]]
) -> np.ndarray:
    """
    Helper function to create contact sequence from phase descriptions.

    Args:
        total_duration: Total trajectory duration
        dt: Time step
        phases: Dictionary of phases with timing and contact patterns

    Returns:
        Contact sequence array (4 x N) where N = total_duration/dt
    """
    horizon = int(total_duration / dt)
    contact_sequence = np.ones((4, horizon))  # Default: all feet in contact

    for _, phase_data in phases.items():
        start_step = int(phase_data["start_time"] / dt)
        duration_steps = int(phase_data["duration"] / dt)
        end_step = min(start_step + duration_steps, horizon)

        for foot in range(4):
            contact_sequence[foot, start_step:end_step] = phase_data["contact_pattern"][
                foot
            ]

    return contact_sequence


def create_phase_sequence(
    phase_list: list[tuple[str, float, list[int]]],
    mpc_duration: float,
    mpc_dt: float,
) -> np.ndarray:
    """
    Helper to create contact sequence from a simple phase list.

    Args:
        phase_list: List of (phase_name, duration, [FL,FR,RL,RR]) tuples
        mpc_duration: Total MPC duration
        mpc_dt: MPC time step

    Returns:
        Contact sequence array
    """
    horizon = int(mpc_duration / mpc_dt)
    contact_sequence = np.ones((4, horizon))

    current_step = 0
    for _, duration, contact_pattern in phase_list:
        duration_steps = int(duration / mpc_dt)
        end_step = min(current_step + duration_steps, horizon)

        for foot in range(4):
            contact_sequence[foot, current_step:end_step] = contact_pattern[foot]

        current_step = end_step
        if current_step >= horizon:
            break

    return contact_sequence


def classify_contact_pattern(pattern: tuple[float, ...]) -> str:
    """Classify a contact pattern into a phase type."""
    num_contacts = sum(pattern)

    if num_contacts == 4:
        return "stance"
    elif num_contacts == 0:
        return "flight"
    elif num_contacts == 2:
        if pattern[0] == pattern[1] and pattern[2] == pattern[3]:
            return "trot" if pattern[0] != pattern[2] else "pace"
        else:
            return "diagonal"
    elif num_contacts == 3:
        return "tripod"
    elif num_contacts == 1:
        return "single_foot"
    else:
        return "unknown"


def analyze_contact_phases(
    contact_sequence: np.ndarray | None, mpc_dt: float
) -> list[dict[str, Any]]:
    """Analyze the contact sequence to identify distinct phases."""
    if contact_sequence is None:
        return []

    phases = []
    current_pattern = None
    phase_start = 0

    for step in range(contact_sequence.shape[1]):
        pattern = tuple(contact_sequence[:, step])

        if pattern != current_pattern:
            if current_pattern is not None:
                phases.append(
                    {
                        "start_time": phase_start * mpc_dt,
                        "duration": (step - phase_start) * mpc_dt,
                        "contact_pattern": list(current_pattern),
                        "phase_type": classify_contact_pattern(current_pattern),
                    }
                )
            current_pattern = pattern
            phase_start = step

    # Add final phase
    if current_pattern is not None:
        phases.append(
            {
                "start_time": phase_start * mpc_dt,
                "duration": (contact_sequence.shape[1] - phase_start) * mpc_dt,
                "contact_pattern": list(current_pattern),
                "phase_type": classify_contact_pattern(current_pattern),
            }
        )

    return phases
