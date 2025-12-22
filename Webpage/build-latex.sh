#!/bin/bash
# Build LaTeX diagrams for the Quarto website

set -e  # Exit on error

echo "Building LaTeX diagrams..."

# Compile unet.tex
if [ ! -f pytorch/segmentation/unet.png ]; then
  echo "Compiling unet.tex..."
  cd latex
  pdflatex -interaction=nonstopmode unet.tex
  cd ..

  # Convert PDF to PNG for web display
  echo "Converting unet.pdf to PNG..."
  pdftoppm -png -r 300 -singlefile latex/unet.pdf pytorch/segmentation/unet

  echo "LaTeX diagrams built successfully!"
else
  echo "File unet.png already exists, no need to regenerate it"
fi
