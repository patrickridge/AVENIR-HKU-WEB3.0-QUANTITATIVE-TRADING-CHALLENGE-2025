#!/bin/bash
# Run the trading strategy.
# Set OMS_URL, OMS_ACCESS_TOKEN, and MODE in your environment or in a .env file:
#   cp ../.env.example ../.env
#   source ../.env
#   bash run.sh

if [ -z "$OMS_ACCESS_TOKEN" ]; then
  echo "Error: OMS_ACCESS_TOKEN is not set. Copy .env.example to .env and fill in your credentials."
  exit 1
fi

if [ -z "$OMS_URL" ]; then
  echo "Error: OMS_URL is not set. Copy .env.example to .env and fill in your credentials."
  exit 1
fi

exec python demo.py
