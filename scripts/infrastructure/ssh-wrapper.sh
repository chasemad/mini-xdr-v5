#!/bin/bash
# SSH Wrapper for Cursor Terminal Compatibility
# This script ensures SSH commands work properly from Cursor's integrated terminal

# Load SSH keys if not already loaded
if ! ssh-add -l >/dev/null 2>&1; then
    echo "Loading SSH keys..."
    ssh-add ~/.ssh/xdrops_id_ed25519 >/dev/null 2>&1
fi

# Export SSH environment variables
export SSH_AUTH_SOCK="/private/tmp/com.apple.launchd.5gL8fnBCeQ/Listeners"

# Execute the SSH command with proper parameters
exec ssh "$@"
