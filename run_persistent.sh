#!/bin/bash
# run_persistent.sh

echo "Select launch mode:"
echo "1) Nohup (Headless Background) - Launches app in background, no GUI window will appear."
echo "2) Xpra (Persistent GUI) - Launches GUI in a detach-able session."
read -p "Enter choice [1 or 2]: " choice

if [ "$choice" == "1" ]; then
    nohup python run_app.py > app_log.txt 2>&1 &
    echo "App launched in background. Check app_log.txt."
elif [ "$choice" == "2" ]; then
    echo "Starting Xpra session on :100..."
    xpra start :100 --start-child="python run_app.py"
    echo "Launched. Attach from your client using: xpra attach ssh:user@host:100"
else
    echo "Invalid choice."
fi