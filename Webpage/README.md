# Deep Learning Lectures - Quarto Website

This is a Quarto conversion of the original pandoc-based documentation for deep learning lectures.

## Structure

The project contains:

- **Lab sessions**: Practical exercises on various deep learning topics
  - Keras MNIST classification
  - PyTorch FashionMNIST 
  - Keras CIFAR classification
  - PyTorch object detection
  - PyTorch segmentation
  - PyTorch automatic speech recognition (ASR)
  - PyTorch generative adversarial networks (GAN)

- **Resources**: Additional documentation
  - Argparse tutorial
  - Cluster usage guide
  - How-to guides
  - FAQ

## Building the Website

To preview the website locally:

```bash
quarto preview
```

To render the website:

```bash
quarto render
```

The rendered site will be in the `_site/` directory.

## Converting from Pandoc

This project was converted from a pandoc-based build system. Key conversions include:

1. YAML frontmatter delimiter changed from `...` to `---`
2. Code blocks: ```` ```{.sourceCode .python} ```` → ```` ```{python} ````
3. Callout boxes: `<div class="w3-card w3-red">` → `::: {.callout-important}`
4. File extension: `.md` → `.qmd`

## Requirements

- [Quarto](https://quarto.org/) 1.3 or later

## License

See the parent LICENSE.txt file.
