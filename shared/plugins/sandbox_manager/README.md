
# Sandbox Command (`sandbox`)

This document provides user-facing documentation for the `sandbox` command.

## Overview

The `sandbox` command is a runtime tool for managing the filesystem paths that the `jaato` agent is allowed to access during a specific session. It allows you to grant temporary permissions or create temporary blocks for paths that are outside the standard workspace.

This system operates on a three-tiered configuration model.

## Configuration Levels

The effective sandbox is determined by merging configurations from three sources. The `sandbox list` command will show you the complete, unified view.

### 1. Global Configuration (Lowest Precedence)

This file is for your personal, user-specific settings that apply to all your projects.

- **Path:** `~/.jaato/sandbox_paths.json`
- **Usage:** Manually edit this file to add paths that you want to be available in every `jaato` session.

### 2. Workspace Configuration

This file is for project-specific settings and can be shared with your team.

- **Path:** `<workspace_root>/.jaato/sandbox.json`
- **Usage:** Manually edit this file to define allowed paths that are relevant to this specific project.

### 3. Session Configuration (Highest Precedence)

This is a temporary configuration that applies only to the current, active session. The `sandbox` command exclusively reads from and writes to this level.

- **Path:** `<workspace_root>/.jaato/sessions/<session_id>/sandbox.json`
- **Usage:** Use the `sandbox add` and `sandbox remove` commands to modify this file at runtime.

## Commands

### `sandbox list`

Displays a complete, read-only list of all currently active sandbox paths from all three configuration levels.

- **Example Output:**
  ```
  Effective Sandbox Paths for this Session:

  [ALLOW] /tmp/temp_data_for_this_session (Session)
  [ALLOW] /var/www/my_project/assets    (Workspace)
  [DENY]  /home/user/project_assets     (Blocked in Session, allowed by Workspace)
  [ALLOW] /opt/company_tools              (Global)
  ```

### `sandbox add <path>`

Grants access to a new path for the current session only.

- **Action:** Adds the specified `<path>` to the `allowed_paths` list in the session's `sandbox.json` file.

### `sandbox remove <path>`

Temporarily blocks access to a path for the current session, even if it is allowed by the Global or Workspace configuration.

- **Action:** Adds the specified `<path>` to the `denied_paths` list in the session's `sandbox.json` file. This acts as a session-level override.

