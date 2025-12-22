# Conversion Summary: Pandoc to Quarto

## Overview

Successfully converted the deep learning lectures documentation from a pandoc-based build system to a modern Quarto website project.

## Project Structure

```
Webpage-quarto/
├── _quarto.yml          # Main Quarto configuration
├── _site/               # Generated website (gitignored)
├── .quarto/             # Quarto cache (gitignored)
├── *.qmd                # 12 Quarto markdown files (converted from .md)
├── data/                # Images and assets (copied from original)
├── biblio.bib           # Bibliography database
├── apsa.csl             # Citation style
├── styles.css           # Custom CSS styling
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## Conversion Steps Performed

### 1. Project Setup
- Created `_quarto.yml` with website configuration
- Set up navigation with navbar and sidebar
- Configured HTML output with table of contents, code highlighting, and custom CSS
- Disabled code execution (`execute: enabled: false`) since this is documentation, not an executable notebook

### 2. Content Migration
- Copied all 12 markdown files (`.md` → `.qmd`):
  - index.qmd (home page)
  - 7 lab session files
  - 4 resource files
- Copied entire `data/` directory with images and assets
- Copied `biblio.bib` and `apsa.csl` for bibliography support

### 3. Syntax Conversions

#### YAML Frontmatter
```diff
- keywords: [Deep learning, practicals]
- ...
+ keywords: [Deep learning, practicals]
+ ---
```

#### Code Blocks
```diff
- ``` {.sourceCode .python}
+ ```{python}

- ``` {.python .numberLines}
+ ```{python}

- ``` {.console}
+ ```{bash}
```

#### Callout Boxes
HTML divs converted to Quarto callouts:
```diff
- <div class="w3-card w3-red">
- **Important**: Message here
- </div>
+ ::: {.callout-important}
+ **Important**: Message here
+ :::
```

Mappings:
- `w3-red` → `.callout-important`
- `w3-sand` → `.callout-note`
- `w3-blue` → `.callout-tip`
- `w3-yellow` → `.callout-warning`
- `w3-green` → `.callout-tip`

#### Image Attributes
Pandoc's `{.bordered}` syntax is compatible with Quarto and retained as-is.

### 4. Styling
Created `styles.css` to preserve visual styling from the original site:
- Bordered images
- Code block styling
- Callout box colors (as fallback)

## Build Results

Successfully rendered with `quarto render`:
- ✅ 12 HTML pages generated
- ✅ Bibliography integration working
- ✅ Navigation functional
- ✅ Code highlighting preserved
- ✅ Images and assets properly linked
- ⚠️  Some warnings about missing "labs" links (expected - external resources)
- ⚠️  Minor fenced div warnings (non-breaking, callouts work correctly)

## Usage

### Preview
```bash
cd Webpage-quarto
quarto preview
```

### Render
```bash
cd Webpage-quarto
quarto render
```

### Output
Generated site is in `_site/` directory

## Advantages of Quarto

1. **Modern tooling**: Active development, excellent documentation
2. **Better syntax**: Native support for callouts, code blocks, cross-references
3. **Flexibility**: Easy to add executable code later if needed
4. **No build dependencies**: No need for pandoc filters, Python preprocessors
5. **Built-in features**: Search, responsive design, themes out of the box
6. **Multi-format**: Can easily export to PDF, EPUB, etc.

## Notes

- Code execution is disabled (suitable for documentation)
- All original content preserved
- Assets organized in `data/` directory
- Bibliography citations maintained
- Custom CSS for backward compatibility
