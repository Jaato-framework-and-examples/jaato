"""Session exporter for saving conversations to YAML format.

Exports conversation sessions to a YAML file that can be replayed
using demo-scripts/run_demo.py.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class SessionExporter:
    """Exports conversation sessions to YAML format for replay."""

    def export_to_yaml(
        self,
        history: List[Any],
        original_inputs: List[Dict[str, Any]],
        filename: str
    ) -> Dict[str, Any]:
        """Export session to a YAML file for replay.

        Args:
            history: List of conversation content objects.
            original_inputs: List of original user input dicts with 'text' and 'local' keys.
            filename: Path to the output YAML file.

        Returns:
            Dict with 'success' bool and 'message' or 'error' string.
        """
        try:
            import yaml
        except ImportError:
            return {
                'success': False,
                'error': "PyYAML is required for export. Install with: pip install pyyaml"
            }

        if not original_inputs:
            return {
                'success': False,
                'error': "No conversation history to export"
            }

        # Extract permission decisions from history, grouped by user turn
        turn_permissions = self._extract_turn_permissions(history)

        # Build steps from original inputs with matched permissions
        final_steps = self._build_export_steps(original_inputs, turn_permissions)

        # Add quit step
        final_steps.append({'type': 'quit', 'delay': 0.08})

        # Build the YAML document
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        export_data = {
            'name': f'Session Export [{timestamp}]',
            'timeout': 120,
            'steps': final_steps,
        }

        # Write to file
        try:
            with open(filename, 'w') as f:
                yaml.dump(
                    export_data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
            return {
                'success': True,
                'message': f"Session exported to: {filename}",
                'step_count': len(final_steps) - 1  # Exclude quit step
            }
        except IOError as e:
            return {
                'success': False,
                'error': f"Error writing file: {e}"
            }

    def _extract_turn_permissions(self, history: List[Any]) -> List[List[str]]:
        """Extract permission decisions from history, grouped by user turn.

        Args:
            history: List of conversation content objects.

        Returns:
            List of permission lists, one per user turn.
        """
        turn_permissions: List[List[str]] = []
        current_permissions: List[str] = []
        in_user_turn = False

        for content in history:
            role = getattr(content, 'role', None) or 'unknown'
            parts = getattr(content, 'parts', None) or []

            if role == 'user':
                # Check if this is a user text message (starts new turn)
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text = part.text.strip()
                        if text.startswith('[User executed command:'):
                            continue
                        # Save previous turn's permissions and start new turn
                        if in_user_turn:
                            turn_permissions.append(current_permissions)
                        current_permissions = []
                        in_user_turn = True

            elif role == 'model':
                # Collect permission data from function responses
                for part in parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        fr = part.function_response
                        response = getattr(fr, 'response', {})
                        if isinstance(response, dict):
                            perm = response.get('_permission')
                            if perm:
                                decision = perm.get('decision', '')
                                method = perm.get('method', '')
                                perm_value = self._map_permission_to_yaml(decision, method)
                                current_permissions.append(perm_value)

        # Don't forget the last turn's permissions
        if in_user_turn:
            turn_permissions.append(current_permissions)

        return turn_permissions

    def _build_export_steps(
        self,
        original_inputs: List[Dict[str, Any]],
        turn_permissions: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Build export steps from original inputs with matched permissions.

        Args:
            original_inputs: List of original user input dicts.
            turn_permissions: List of permission lists per turn.

        Returns:
            List of step dicts for YAML export.
        """
        final_steps = []
        model_turn_index = 0  # Track index for model-bound prompts only

        for input_entry in original_inputs:
            user_input = input_entry["text"]
            is_local = input_entry["local"]

            if is_local:
                # Local commands (plugin commands like "plan") don't need permissions
                final_steps.append({
                    'type': user_input,
                    'local': True,
                })
            else:
                # Model-bound prompts may have permissions
                perms = turn_permissions[model_turn_index] if model_turn_index < len(turn_permissions) else []
                model_turn_index += 1

                # Determine permission value
                permission = self._determine_permission(perms)

                final_steps.append({
                    'type': user_input,
                    'permission': permission,
                })

        return final_steps

    def _determine_permission(self, perms: List[str]) -> str:
        """Determine the most appropriate permission value for a turn.

        Args:
            perms: List of permission values for the turn.

        Returns:
            Single permission value ('y', 'n', 'a', 'never').
        """
        if not perms:
            return 'y'  # Default

        # Use the most permissive permission granted
        # Priority: 'a' (always) > 'y' (yes) > 'n' (no)
        if 'a' in perms:
            return 'a'
        elif 'y' in perms or 'once' in perms:
            return 'y'
        elif 'n' in perms:
            return 'n'
        elif 'never' in perms:
            return 'never'

        return 'y'

    def _map_permission_to_yaml(self, decision: str, method: str) -> str:
        """Map permission decision/method to YAML permission value.

        Args:
            decision: 'allowed' or 'denied'.
            method: 'user', 'remembered', 'whitelist', etc.

        Returns:
            YAML permission value: 'y', 'n', 'a', 'never', 'once'.
        """
        if decision == 'allowed':
            if method == 'remembered':
                return 'a'  # Was 'always' - permission remembered
            elif method == 'whitelist':
                return 'a'  # Auto-approved, use 'always' for replay
            else:
                return 'y'  # User approved this one
        else:  # denied
            if method == 'remembered':
                return 'never'  # Was 'never' - denial remembered
            else:
                return 'n'  # User denied this one
