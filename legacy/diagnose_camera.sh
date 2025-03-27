#!/bin/bash

echo "===== Camera Diagnostic Tool ====="
echo "Running as user: $(whoami)"
echo "Date: $(date)"
echo

echo "===== System Information ====="
uname -a
echo

echo "===== Checking for video devices ====="
ls -la /dev/video*
echo

echo "===== Checking video device permissions ====="
ls -la /dev/video0
echo

echo "===== Checking for processes using the camera ====="
sudo lsof /dev/video* 2>/dev/null || echo "No processes found using video devices"
echo

echo "===== Checking for Python processes ====="
ps aux | grep python | grep -v grep
echo

echo "===== Checking camera with v4l2 ====="
v4l2-ctl --list-devices
echo

echo "===== Checking camera capabilities ====="
v4l2-ctl --all
echo

echo "===== Checking libcamera processes ====="
ps aux | grep libcamera | grep -v grep || echo "No libcamera processes found"
echo

echo "===== Checking for camera hardware ====="
vcgencmd get_camera || echo "vcgencmd not available"
echo

echo "===== Diagnostic complete ====="
echo "If you're still having issues, try rebooting the Raspberry Pi"
echo "with: sudo reboot" 