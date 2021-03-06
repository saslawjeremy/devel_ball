#!/bin/bash

# Get year
YEAR=$1

# Exit on any error
set -e

# Exit on SIGINT
trap "exit" INT

# Execute full year
#python3 load_models.py --gamedates --years $YEAR
#python3 load_models.py --games --years $YEAR
#python3 post_process.py --draftkings --years $YEAR
python3 post_process.py --official-seasons --years $YEAR
python3 post_process.py --team-seasons --years $YEAR
python3 post_process.py --player-seasons --years $YEAR
python3 create_pandas.py --years $YEAR
