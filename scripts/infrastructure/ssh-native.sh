#!/bin/bash
# SSH via Native Terminal for Cursor Compatibility
# This script executes SSH commands through macOS Terminal app to avoid networking issues

if [ $# -eq 0 ]; then
    echo "Usage: $0 <ssh_arguments>"
    exit 1
fi

# Create a temporary script
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
ssh $@
echo "Exit code: \$?"
EOF

chmod +x "$TEMP_SCRIPT"

# Execute via Terminal and capture output
osascript << EOF
tell application "Terminal"
    activate
    set newWindow to do script "$TEMP_SCRIPT"
    
    -- Wait for the command to complete
    repeat
        delay 0.5
        if not busy of newWindow then exit repeat
    end repeat
    
    -- Get the output
    set output to contents of newWindow
    
    -- Close the window
    close newWindow
    
    return output
end tell
EOF

# Clean up
rm -f "$TEMP_SCRIPT"
