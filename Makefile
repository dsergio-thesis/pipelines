
# Makefile

ENV_PREFIX := /home/dsergio/conda/envs/astroos-pipelines-0.1.0-py311

clean-rsp-env:
	mamba env remove -p $(ENV_PREFIX) -y || true

clean:
	rm -rf _pipelines/* data/* log/* /tmp/pipelines_exports/



