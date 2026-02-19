"""SDK-wide constants shared between server and client code."""

# Sentinel prefix for pre-rendered lines (e.g. mermaid diagram half-block art).
# Lines with this prefix have already been rendered with precise ANSI positioning
# and must NOT be re-wrapped by the output buffer -- wrapping breaks pixel-aligned
# half-block characters.  Null bytes never appear in normal model text output.
PRERENDERED_LINE_PREFIX = "\x00\x01PRE\x00"
