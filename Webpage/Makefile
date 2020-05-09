


OUTPUT_DIR=../Site
SRC = $(wildcard *.md)
HTML=$(addprefix $(OUTPUT_DIR)/, $(SRC:.md=.html))

.PHONY: copyextra
copyextra: 
	@cp Makefile $(OUTPUT_DIR)
	@cp biblio.bib $(OUTPUT_DIR)
	@cp -r templates $(OUTPUT_DIR)
	@cp -r data $(OUTPUT_DIR)

all: outputdirs $(HTML) copyextra

outputdirs:
	mkdir -p $(OUTPUT_DIR)/md-sources

clean:
	rm -rf $(OUTPUT_DIR)

$(HTML): | $(OUTPUT_DIR)

css:
	pygmentize -f html -S friendly -a .highlight > ./templates/pygmentize.css

$(OUTPUT_DIR)/%.html: %.md
	pandoc -r markdown+tex_math_dollars+simple_tables+table_captions+yaml_metadata_block+fenced_code_attributes+implicit_figures -w html --mathjax --standalone --table-of-contents --toc-depth=3 --template=./templates/html.template --filter ./templates/pandocfilter-pygments.py --filter pandoc-citeproc --csl=./templates/apsa.csl --bibliography=biblio.bib  -M date="`LC_TIME=en_US.UTF-8 date "+%B %e, %Y %H:%M"`" -M link-citations=true -M source=$< -o $@ $<
	cp $< $(OUTPUT_DIR)/md-sources/


install-deps:
	sudo apt install pandoc pandoc-data pandoc-citeproc python3-pygments python3-pandocfilters

