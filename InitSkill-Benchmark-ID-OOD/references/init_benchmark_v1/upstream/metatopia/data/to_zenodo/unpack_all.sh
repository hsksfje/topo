#!/bin/bash

# 1. Check if 7zip is installed
if ! command -v 7z &> /dev/null; then
    echo "Error: 7z command not found."
    exit 1
fi

echo "Starting bulk extraction..."

# 2. Loop through all .7z files
for f in *.7z; do
    # Check if file exists
    [ -e "$f" ] || continue

    echo "----------------------------------------"
    echo "Processing: $f"
    
    # 3. Extract contents directly to current folder
    # We REMOVED the -o flag.
    # This relies on the archive having its own internal folder (e.g. 'train/')
    7z x "$f" -bsp1 -aos
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $f unpacked."
    else
        echo "❌ Error: Failed to unpack $f"
    fi
done

echo "----------------------------------------"
echo "All tasks finished."
