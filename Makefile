
# Makefile

CWD := $(shell pwd)

# run install.sh
all: install

install:
	@echo "Installing astroos-pipelines in $(CWD)"
	@zsh install.sh 

clean:
	rm -rf _pipelines/* log/* /tmp/pipelines_exports/



