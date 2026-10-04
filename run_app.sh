#!/usr/bin/env bash
# Quick Launcher for Investor Emotions & Stock Market Behaviour Application
cd "$(dirname "$0")"
echo "Launching Investor Emotions & Stock Market Behaviour Dashboard..."
echo "Opening in your default browser..."
streamlit run application/app.py
