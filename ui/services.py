"""Cached access to engines and the example catalogue.

Both are cached so that Streamlit reruns - switching a view, expanding a
section - never reload a model or rescan the catalogue. Inference itself is
never cached here: it runs only when the user presses Analyze.

Engines are cached PER DOMAIN, keyed by name. Asking for one domain never
constructs another, so registering the environment engine did not make the app
load two models: a screen requests its domain and only that checkpoint is read.
The default stays built_environment, which is the domain the current screens
are built around.
"""
import streamlit as st

from src.common import examples as example_catalogue
from src.common import pair_input
from src.core import registry

DEFAULT_DOMAIN = "built_environment"

#: Retained name for the default domain. Prefer DEFAULT_DOMAIN.
ENGINE_NAME = DEFAULT_DOMAIN


@st.cache_resource(show_spinner=False)
def get_engine(name=DEFAULT_DOMAIN):
    """The domain engine, loaded once per session per domain.

    No checkpoint is passed: each engine resolves its own configured default.
    """
    return registry.get(name)


def available_domains():
    """Domains that have both an engine and an input adapter."""
    return [name for name in registry.available() if name in pair_input.domains()]


@st.cache_data(show_spinner=False)
def domain_card(name):
    """Display title and blurb for a domain, WITHOUT constructing its engine.

    Home must not load two neural networks just to label two buttons, so this
    reads the domain's model card - the same declarations metadata is built
    from - rather than calling get_engine(). A domain whose card cannot be
    imported still gets a usable title instead of breaking the screen.
    """
    import importlib

    title = name.replace("_", " ").title()
    card = {"domain": name, "display_name": title, "task": "", "description": "",
            "bands": []}
    try:
        module = importlib.import_module(f"src.domains.{name}.model_card")
    except Exception:                                            # noqa: BLE001
        return card
    card["display_name"] = getattr(module, "DISPLAY_NAME", title)
    card["task"] = getattr(module, "TASK", "")
    card["description"] = getattr(module, "DESCRIPTION", "")
    # The bands the domain declares, so Home can state an input requirement
    # without constructing the engine that enforces it.
    spec = getattr(module, "INPUT_SPEC", None)
    card["bands"] = list(getattr(spec, "bands", ()) or ())
    return card


def prepare_pair(domain, before, after, input_spec=None):
    """Read a before/after pair the way `domain` requires.

    The screens call this instead of assuming RGB; the dispatch and every
    domain-specific rule live in src/common/pair_input.py, not in the UI.
    """
    return pair_input.prepare(domain, before, after, input_spec)


@st.cache_data(show_spinner=False)
def list_examples(domain):
    """Catalogue entries that resolve on this machine, as plain dicts."""
    return [e.to_dict() for e in example_catalogue.available(domain=domain)]


def example_from_dict(d):
    return example_catalogue.Example(**d)
