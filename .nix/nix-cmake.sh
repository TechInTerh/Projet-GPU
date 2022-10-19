#!/usr/bin/env sh


# Use the cmakeFlags set by Nix - this doesn't work with --build
FLAGS=$(echo "$@" | grep -e '--build' > /dev/null || echo "$cmakeFlags")
echo "$(dirname "$0")"/nix-run.sh cmake ${FLAGS:+"$FLAGS"} "$@"
"$(dirname "$0")"/nix-run.sh cmake ${FLAGS:+"$FLAGS"} "$@"