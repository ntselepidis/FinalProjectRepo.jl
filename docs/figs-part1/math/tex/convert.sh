#!/bin/bash

set -euo pipefail

# Compile all tex files
latexmk -pdf *.tex

# Convert pdfs to pngs
for f in *.pdf; do
    fname="$( basename "$f" | sed 's/\..*//g' )"
    convert -density 600 -background white -alpha off ${fname}.pdf -resize 20% ${fname}.png
done

# Clean directory
latexmk -C
