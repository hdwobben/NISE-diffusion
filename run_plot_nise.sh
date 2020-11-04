#!/bin/bash

CLEAR='\033[0m'
RED='\033[0;31m'
python_interpreter="/home/hans/anaconda2/envs/latest/bin/python"

function usage() {
  if [ -n "$1" ]; then
    echo -e "${RED}$1${CLEAR}\n";
  fi
  echo "Usage: $0 [-r] [-- passed-args]"
  echo "  -r, --remove   Remove intermediate data file after running"
  echo "  --             Flags to be passed along to the calculation"
  echo ""
  echo "Example: $0 -r -- -x -q -p params.json"
  exit 1
}

# parse params
while [[ "$#" > 0 ]]; do case $1 in
  -r|--remove) REMOVE=1; shift;;
  --) shift; break;;
  *) usage "Unknown parameter passed: $1";;
esac; done

if [ -z "$REMOVE" ]; then 
  OUTPUT=$(./build/Release/bin/nise_diff "$@" | tee /dev/tty | tail -1)
  eval "$python_interpreter" plot_msd.py $OUTPUT;
else 
  ./build/Release/bin/nise_diff "$@" -o out.tmp
  eval "$python_interpreter" plot_msd.py out.tmp
fi