"""Pane manager for vertical split-pane output layout.

Manages pane layout state separately from AgentRegistry (which owns agent data).
PaneManager owns the spatial layout: which agents are in which pane, which pane
has focus, and the prompt_toolkit rendering pipeline for each pane slot.

Pane slots are pre-allocated (always 4) to avoid dynamic prompt_toolkit object
lifecycle issues. Only `_active_pane_count` of them are visible at any time.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.dimension import Dimension

logger = logging.getLogger(__name__)

# Minimum width per pane in columns
MIN_PANE_WIDTH = 20


@dataclass
class PaneSlot:
    """A single pane slot with its own prompt_toolkit rendering pipeline.

    Each slot holds the prompt_toolkit objects needed to display one pane:
    a Buffer for plain text, a dict for style fragments, a StyledOutputProcessor,
    a ScrollableBufferControl, and a Window. These are pre-allocated in _build_app()
    and wired into the slot via init_slot().

    Attributes:
        pane_index: Position of this slot (0-3).
        agent_ids: Agents assigned to this pane.
        visible_agent_id: Currently displayed agent in this pane.
        pt_buffer: prompt_toolkit read-only buffer for plain text.
        line_fragments: Per-line style fragments for StyledOutputProcessor.
        control: ScrollableBufferControl for display.
        processor: StyledOutputProcessor for styling.
        window: prompt_toolkit Window for layout.
        dirty: Whether this pane needs re-render.
    """
    pane_index: int
    agent_ids: List[str] = field(default_factory=list)
    visible_agent_id: Optional[str] = None
    pt_buffer: Optional[Buffer] = None
    line_fragments: Optional[Dict[int, list]] = None
    control: object = None  # ScrollableBufferControl
    processor: object = None  # StyledOutputProcessor
    window: Optional[Window] = None
    dirty: bool = True


class PaneManager:
    """Manages vertical split-pane layout for the TUI output panel.

    Owns the spatial layout of agents across panes. The AgentRegistry owns
    agent data (buffers, history); PaneManager maps agents to visual pane
    positions.

    Key behaviors:
    - Pre-allocates MAX_PANES slots; only _active_pane_count are visible.
    - Main agent is pinned to pane 0 and cannot be moved.
    - Empty panes (except pane 0) auto-collapse from the right.
    - New agents are assigned to the focused pane by default.
    """

    MAX_PANES = 4

    def __init__(self):
        """Initialize with 4 pre-allocated slots, 1 active."""
        self._slots: List[PaneSlot] = [PaneSlot(pane_index=i) for i in range(self.MAX_PANES)]
        self._active_pane_count: int = 1
        self._focused_pane: int = 0

    def init_slot(
        self,
        index: int,
        pt_buffer: Buffer,
        line_fragments: Dict[int, list],
        processor: object,
        control: object,
        window: Window,
    ) -> None:
        """Wire prompt_toolkit objects into a pre-allocated slot.

        Called during _build_app() to set up the rendering pipeline for each slot.

        Args:
            index: Slot index (0-3).
            pt_buffer: prompt_toolkit Buffer for this pane.
            line_fragments: Dict for per-line style fragments.
            processor: StyledOutputProcessor instance.
            control: ScrollableBufferControl instance.
            window: prompt_toolkit Window instance.
        """
        slot = self._slots[index]
        slot.pt_buffer = pt_buffer
        slot.line_fragments = line_fragments
        slot.processor = processor
        slot.control = control
        slot.window = window

    def get_slot(self, index: int) -> PaneSlot:
        """Get a pane slot by index.

        Args:
            index: Slot index (0-3).

        Returns:
            The PaneSlot at the given index.
        """
        return self._slots[index]

    def get_focused_slot(self) -> PaneSlot:
        """Get the currently focused pane slot.

        Returns:
            The PaneSlot that has focus.
        """
        return self._slots[self._focused_pane]

    # ─────────────────────────────────────────────────────────────────────────
    # Pane operations
    # ─────────────────────────────────────────────────────────────────────────

    def split_pane(self, terminal_width: int) -> bool:
        """Add a vertical pane to the right.

        Refuses if already at MAX_PANES or terminal too narrow.

        Args:
            terminal_width: Current terminal width in columns.

        Returns:
            True if a pane was added.
        """
        if self._active_pane_count >= self.MAX_PANES:
            return False
        # Check width: each pane needs at least MIN_PANE_WIDTH columns
        new_count = self._active_pane_count + 1
        if terminal_width // new_count < MIN_PANE_WIDTH:
            return False
        self._active_pane_count = new_count
        return True

    def join_pane(self) -> bool:
        """Remove the rightmost pane, moving its agents to the previous pane.

        Cannot remove pane 0 (minimum 1 pane).

        Returns:
            True if a pane was removed.
        """
        if self._active_pane_count <= 1:
            return False
        removed_idx = self._active_pane_count - 1
        target_idx = removed_idx - 1
        removed_slot = self._slots[removed_idx]
        target_slot = self._slots[target_idx]
        # Move agents from removed pane to target
        for agent_id in removed_slot.agent_ids:
            if agent_id not in target_slot.agent_ids:
                target_slot.agent_ids.append(agent_id)
        # If target pane had no visible agent, pick one from the moved agents
        if not target_slot.visible_agent_id and target_slot.agent_ids:
            target_slot.visible_agent_id = target_slot.agent_ids[0]
        # Clear removed slot
        removed_slot.agent_ids.clear()
        removed_slot.visible_agent_id = None
        removed_slot.dirty = True
        target_slot.dirty = True
        self._active_pane_count -= 1
        # Move focus if it was on the removed pane
        if self._focused_pane >= self._active_pane_count:
            self._focused_pane = self._active_pane_count - 1
        return True

    def move_agent(self, agent_id: str, terminal_width: int) -> bool:
        """Move an agent to the next pane cyclically.

        Main agent (pane 0) is pinned and cannot be moved.
        Cycles: current pane -> next pane, wrapping from last active to pane 1
        (skipping pane 0). If the target pane doesn't exist and count < MAX,
        auto-splits first.

        Args:
            agent_id: The agent to move.
            terminal_width: Current terminal width for auto-split check.

        Returns:
            True if the agent was moved.
        """
        if agent_id == "main":
            return False
        source_idx = self.get_pane_for_agent(agent_id)
        if source_idx is None:
            return False
        # Calculate target: next pane, wrapping to 1 (not 0)
        if self._active_pane_count <= 1:
            # Only one pane - try to auto-split
            if not self.split_pane(terminal_width):
                return False
        target_idx = source_idx + 1
        if target_idx >= self._active_pane_count:
            target_idx = 1 if self._active_pane_count > 1 else 0
        if target_idx == source_idx:
            return False
        # If target doesn't exist yet and we can split, do so
        if target_idx >= self._active_pane_count:
            if not self.split_pane(terminal_width):
                return False
        source_slot = self._slots[source_idx]
        target_slot = self._slots[target_idx]
        # Move agent
        if agent_id in source_slot.agent_ids:
            source_slot.agent_ids.remove(agent_id)
        if agent_id not in target_slot.agent_ids:
            target_slot.agent_ids.append(agent_id)
        # Update visible agents
        if source_slot.visible_agent_id == agent_id:
            source_slot.visible_agent_id = source_slot.agent_ids[0] if source_slot.agent_ids else None
        target_slot.visible_agent_id = agent_id
        source_slot.dirty = True
        target_slot.dirty = True
        # Move focus to target pane
        self._focused_pane = target_idx
        return True

    def auto_collapse_empty(self) -> None:
        """Remove empty panes from the right (except pane 0).

        Scans from rightmost active pane inward, collapsing any empty pane.
        """
        while self._active_pane_count > 1:
            rightmost = self._slots[self._active_pane_count - 1]
            if not rightmost.agent_ids:
                rightmost.visible_agent_id = None
                rightmost.dirty = True
                self._active_pane_count -= 1
                if self._focused_pane >= self._active_pane_count:
                    self._focused_pane = self._active_pane_count - 1
            else:
                break

    # ─────────────────────────────────────────────────────────────────────────
    # Agent-pane mapping
    # ─────────────────────────────────────────────────────────────────────────

    def assign_agent(self, agent_id: str, pane_index: int) -> None:
        """Assign an agent to a pane.

        If the pane_index is beyond active count, assigns to the last active pane.

        Args:
            agent_id: The agent to assign.
            pane_index: Target pane index.
        """
        # Clamp to active range
        idx = min(pane_index, self._active_pane_count - 1)
        slot = self._slots[idx]
        if agent_id not in slot.agent_ids:
            slot.agent_ids.append(agent_id)
        # If pane has no visible agent, show this one
        if not slot.visible_agent_id:
            slot.visible_agent_id = agent_id
        slot.dirty = True

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from its pane.

        Args:
            agent_id: The agent to remove.
        """
        for slot in self._slots[:self._active_pane_count]:
            if agent_id in slot.agent_ids:
                slot.agent_ids.remove(agent_id)
                if slot.visible_agent_id == agent_id:
                    slot.visible_agent_id = slot.agent_ids[0] if slot.agent_ids else None
                slot.dirty = True
                break

    def get_pane_for_agent(self, agent_id: str) -> Optional[int]:
        """Find which pane contains an agent.

        Args:
            agent_id: The agent to find.

        Returns:
            Pane index, or None if agent is not assigned to any pane.
        """
        for slot in self._slots[:self._active_pane_count]:
            if agent_id in slot.agent_ids:
                return slot.pane_index
        return None

    def set_visible_agent(self, pane_index: int, agent_id: str) -> None:
        """Set which agent is displayed in a pane.

        Args:
            pane_index: The pane to update.
            agent_id: The agent to display.
        """
        if pane_index < self._active_pane_count:
            slot = self._slots[pane_index]
            if agent_id in slot.agent_ids:
                slot.visible_agent_id = agent_id
                slot.dirty = True

    # ─────────────────────────────────────────────────────────────────────────
    # Focus
    # ─────────────────────────────────────────────────────────────────────────

    def set_focus(self, pane_index: int) -> None:
        """Set which pane has focus.

        Args:
            pane_index: The pane index to focus (clamped to active range).
        """
        self._focused_pane = min(pane_index, self._active_pane_count - 1)

    @property
    def focused_pane(self) -> int:
        """Index of the currently focused pane."""
        return self._focused_pane

    @property
    def active_pane_count(self) -> int:
        """Number of currently visible panes."""
        return self._active_pane_count

    # ─────────────────────────────────────────────────────────────────────────
    # Layout queries
    # ─────────────────────────────────────────────────────────────────────────

    def get_active_slots(self) -> List[PaneSlot]:
        """Get only the visible pane slots.

        Returns:
            List of PaneSlot objects for active panes (length = _active_pane_count).
        """
        return self._slots[:self._active_pane_count]

    def mark_all_dirty(self) -> None:
        """Mark all active panes as needing re-render."""
        for slot in self._slots[:self._active_pane_count]:
            slot.dirty = True
