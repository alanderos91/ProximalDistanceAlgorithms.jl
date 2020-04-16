#!/usr/local/bin/bash

parent=$(basename "$(dirname "$PWD")")
dir=$(basename "$PWD")
target1="ProximalDistanceAlgorithms"
target2="experiments"

if [[ ("$parent" == "$target1") && ("$dir" == "$target2") ]]; then
    echo "Cleaning $parent/$dir: Proceed? [y/Y]"
    read confirm
    confirm=${confirm^^}
    if [[ "$confirm" == "Y" ]]; then
        for dir in ./*; do
            if [[ -d "$dir" ]]; then
                echo "  cleaning $dir..."
                find "./$dir" -type f -delete
            fi
        done
        echo "Complete."
    fi
fi
