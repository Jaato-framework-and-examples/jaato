"""The caps on bytes the daemon writes into a workspace ON BEHALF of a
client or a session.

One definition, read by every write path: the WS file-staging verbs
(``StageFilesRequest`` and the legacy inline ``staged_files``, in
``websocket.py``) and the cross-workspace copy a group message makes into
its target's inbox (``session_inbox.store_file``, design §4.5).  Two
modules each holding "10 MB per file, 50 MB per batch" is how the two
drift apart, and a copy bounded more loosely than a stage would be a way
to put more bytes in a workspace by messaging a session there than by
staging into it.

Stdlib-only, so :mod:`.session_inbox` can import it and stay importable
from the daemon's listing path.
"""

#: Largest single file, in bytes: protects against one huge upload (or
#: copy) tying up the daemon.
STAGE_PER_FILE_LIMIT = 10 * 1024 * 1024   # 10 MB

#: Largest batch -- one staging call, or one message's copied files -- in
#: bytes: protects against many smaller files totalling something huge.
STAGE_TOTAL_LIMIT = 50 * 1024 * 1024      # 50 MB
