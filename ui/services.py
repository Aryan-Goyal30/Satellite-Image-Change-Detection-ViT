"""Cached access to the engine and the example catalogue.

Both are cached so that Streamlit reruns - switching a view, expanding a
section - never reload the model or rescan the catalogue. Inference itself is
never cached here: it runs only when the user presses Analyze.
"""
import streamlit as st

from src.common import examples as example_catalogue
from src.core import registry

ENGINE_NAME = "built_environment"


@st.cache_resource(show_spinner=False)
def get_engine(name=ENGINE_NAME):
    """The domain engine, loaded once per session.

    No checkpoint is passed: the engine resolves its own configured default.
    """
    return registry.get(name)


@st.cache_data(show_spinner=False)
def list_examples(domain):
    """Catalogue entries that resolve on this machine, as plain dicts."""
    return [e.to_dict() for e in example_catalogue.available(domain=domain)]


def example_from_dict(d):
    return example_catalogue.Example(**d)
