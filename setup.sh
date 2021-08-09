#!/bin/bash
# jupyterlabのためにnodejsをinstall
# https://jupyterlab.readthedocs.io/en/stable/user/extensions.html
conda install -c conda-forge nodejs

# table of contents
jupyter labextension install @jupyterlab/toc

# 