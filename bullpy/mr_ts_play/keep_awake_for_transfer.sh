#!/bin/bash
# Keep laptop awake during transfer

echo "=========================================="
echo "Keeping Laptop Awake During Transfer"
echo "=========================================="
echo ""
echo "This will prevent your laptop from sleeping"
echo "Press Ctrl+C to stop and allow sleep again"
echo ""
echo "Starting caffeinate (prevents sleep)..."
echo ""

# Prevent sleep while transfer is running
# -d: prevent display sleep
# -i: prevent idle sleep
# -m: prevent disk sleep
# -s: prevent system sleep
caffeinate -d -i -m -s

echo ""
echo "Caffeinate stopped. Laptop can now sleep normally."


