#!/bin/bash
# Rename all .tiff files in each Camera* subfolder to %08d.tiff, starting from 0 in each folder
for dir in Camera*/; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        n=0
        for f in "$dir"*.tiff; do
            [ -e "$f" ] || continue
            mv -- "$f" "${dir}$(printf "%08d.tiff" $n)"
            n=$((n+1))
        done
    fi
done
