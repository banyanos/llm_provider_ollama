#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

__author__ = "bibow"

__all__ = ["llm_provider_ollama", "OllamaProvider"]
from .llm_provider_ollama import OllamaProvider

#: Discovered by llm_engine's ``llm_provider_*`` package scan when the
#: project is on PYTHONPATH without being pip-installed (bind mount); the
#: ``banyan.llm_providers`` entry points cover installed packages.
PROVIDERS = [OllamaProvider]
