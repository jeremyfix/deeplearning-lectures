


OUTPUT_DIR=../Site
SRC = $(wildcard *.md)
HTML=$(SRC:.md=.html)

all: $(HTML)
	@cp Makefile $(OUTPUT_DIR)
	@cp biblio.bib $(OUTPUT_DIR)
	@cp -r templates $(OUTPUT_DIR)
	@cp -r data $(OUTPUT_DIR)
	@cp *.md $(OUTPUT_DIR)/md-sources


css:
	pygmentize -f html -S friendly -a .highlight > ./templates/pygmentize.css

%.html: %.md
	pandoc -r markdown+tex_math_dollars+simple_tables+table_captions+yaml_metadata_block+fenced_code_attributes -w html --mathjax --standalone --table-of-contents --toc-depth=3 --template=./templates/html.template --filter ./templates/pandocfilter-pygments.py --filter pandoc-citeproc --csl=./templates/apsa.csl --bibliography=biblio.bib  -M date="`LC_TIME=en_US.UTF-8 date "+%B %e, %Y"`" -M link-citations=true -M source=$< -o $(OUTPUT_DIR)/$@ $<


install-deps:
	sudo apt install pandoc pandoc-data pandoc-citeproc python3-pygments python3-pandocfilters

