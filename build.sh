#!/usr/bin/env bash
# build.sh — Render build script
# 1. Install Python dependencies
# 2. Build React frontend

set -o errexit

pip install -r requirements.txt

# Build frontend
cd frontend
npm install
npm run build
cd ..
