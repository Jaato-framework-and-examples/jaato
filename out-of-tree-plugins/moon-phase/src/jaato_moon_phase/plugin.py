"""Moon Phase Calculator Plugin for jaato.

This plugin provides tools to calculate moon phases for any given date.
"""

from datetime import datetime, timezone
import math
from typing import Dict, List, Any


class MoonPhasePlugin:
    """Plugin that calculates moon phases."""

    name = "moon_phase"

    def __init__(self):
        self.precision = 2

    def initialize(self, config: Dict[str, Any]):
        """Initialize plugin with configuration.

        Args:
            config: Configuration dictionary with optional keys:
                - precision: Number of decimal places for illumination (default: 2)
        """
        self.precision = config.get("precision", 2)

    def get_tool_schemas(self) -> List:
        """Return tool schemas for this plugin.

        Returns:
            List of ToolSchema objects declaring available tools.
        """
        # Import from jaato package (external plugin pattern)
        from jaato import ToolSchema

        return [
            ToolSchema(
                name="calculate_moon_phase",
                description="""Calculate the current moon phase for a given date.
                Returns the phase name (New Moon, Waxing Crescent, First Quarter,
                Waxing Gibbous, Full Moon, Waning Gibbous, Last Quarter, Waning Crescent)
                and the illumination percentage. Use this tool when users ask about
                the moon phase, lunar cycle, or what the moon looks like on a specific date.""",
                parameters={
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in YYYY-MM-DD format. If not provided, uses current date."
                        },
                        "include_details": {
                            "type": "boolean",
                            "description": "Include additional astronomical details like age and distance"
                        }
                    },
                    "required": []
                }
            )
        ]

    def get_executors(self) -> Dict[str, Any]:
        """Map tool names to executor functions.

        Returns:
            Dictionary mapping tool names to callable functions.
        """
        return {
            "calculate_moon_phase": self._calculate_moon_phase
        }

    def _calculate_moon_phase(
        self,
        date: str = None,
        include_details: bool = False
    ) -> str:
        """Execute moon phase calculation.

        Args:
            date: Date string in YYYY-MM-DD format. If None, uses current date.
            include_details: Whether to include additional astronomical details.

        Returns:
            String describing the moon phase and illumination.
        """
        try:
            # Parse date or use current
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    return f"Error: Invalid date format '{date}'. Please use YYYY-MM-DD format (e.g., 2024-12-31)"
            else:
                target_date = datetime.now(timezone.utc)

            # Calculate moon phase using astronomical algorithm
            phase_info = self._compute_moon_phase(target_date)

            # Format the result
            result_lines = [
                f"Moon Phase for {target_date.strftime('%Y-%m-%d')}:",
                f"  Phase: {phase_info['phase_name']}",
                f"  Illumination: {phase_info['illumination']:.{self.precision}f}%"
            ]

            if include_details:
                result_lines.extend([
                    f"  Age: {phase_info['age']:.1f} days",
                    f"  Phase Angle: {phase_info['phase_angle']:.1f}°"
                ])

            return "\n".join(result_lines)

        except Exception as e:
            # Never let exceptions propagate - return error message
            return f"Error calculating moon phase: {str(e)}"

    def _compute_moon_phase(self, date: datetime) -> Dict[str, Any]:
        """Compute moon phase using astronomical algorithms.

        Uses a simplified lunar phase algorithm based on the synodic month.

        Args:
            date: Target date for calculation

        Returns:
            Dictionary with phase information
        """
        # Known new moon reference: January 6, 2000, 18:14 UTC
        reference_new_moon = datetime(2000, 1, 6, 18, 14, 0, tzinfo=timezone.utc)

        # Synodic month (average time between new moons)
        synodic_month = 29.53058867  # days

        # Calculate days since reference new moon
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)

        days_since_reference = (date - reference_new_moon).total_seconds() / 86400

        # Calculate position in current lunar cycle
        lunar_age = days_since_reference % synodic_month

        # Phase angle (0° = new moon, 180° = full moon)
        phase_angle = (lunar_age / synodic_month) * 360

        # Calculate illumination percentage
        # Illumination follows a cosine curve
        illumination = (1 - math.cos(math.radians(phase_angle))) / 2 * 100

        # Determine phase name based on angle
        phase_name = self._get_phase_name(phase_angle)

        return {
            "phase_name": phase_name,
            "illumination": illumination,
            "age": lunar_age,
            "phase_angle": phase_angle
        }

    def _get_phase_name(self, phase_angle: float) -> str:
        """Get the name of the moon phase based on angle.

        Args:
            phase_angle: Phase angle in degrees (0-360)

        Returns:
            Name of the moon phase
        """
        # Normalize angle to 0-360
        angle = phase_angle % 360

        # Define phase boundaries
        if angle < 22.5 or angle >= 337.5:
            return "New Moon"
        elif 22.5 <= angle < 67.5:
            return "Waxing Crescent"
        elif 67.5 <= angle < 112.5:
            return "First Quarter"
        elif 112.5 <= angle < 157.5:
            return "Waxing Gibbous"
        elif 157.5 <= angle < 202.5:
            return "Full Moon"
        elif 202.5 <= angle < 247.5:
            return "Waning Gibbous"
        elif 247.5 <= angle < 292.5:
            return "Last Quarter"
        else:  # 292.5 <= angle < 337.5
            return "Waning Crescent"
