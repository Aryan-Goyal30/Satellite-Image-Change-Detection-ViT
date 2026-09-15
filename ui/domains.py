"""Per-domain PRESENTATION facts for the input screen.

What belongs here: which file extensions the uploader offers, and the wording
shown to a user. Nothing else.

What must never come here: band arithmetic, normalisation, reflectance
conversion, model behaviour, or the list of which domains exist. Availability
comes from services.available_domains(), which reads the engine registry - this
module only supplies copy for names that registry already reported, and falls
back to a neutral profile for a name it has never heard of. Adding a domain here
does NOT make it appear in the product.

Band names are NOT written here. `requires` carries a {bands} placeholder that
the screen fills from the engine's own declared InputSpec, so the requirement a
user reads cannot drift from the one the model actually enforces.
"""

#: Presentation profile per domain. Keyed by domain name, but NOT a registry.
PROFILES = {
    "built_environment": {
        # Home card, in the order a card reads: what it finds, what it
        # needs, what else it offers.
        "card_finds": "Construction and other built-up area changes between two dates.",
        "card_needs": ("Two ordinary RGB satellite images of the same place, "
                       "taken at different times."),
        "card_extra": ("Detected changes can also be classified as Construction, "
                       "Demolition, or Uncertain."),
        "purpose": ("Detect construction and other built-up area changes from "
                    "before-and-after RGB satellite images."),
        "requires": "Two RGB images of the same place, taken at different times.",
        "requires_note": "",
        "details": ("Best results come from images that are reasonably aligned "
                    "and comparable. Multispectral and SAR products are not "
                    "supported. Georeferenced GeoTIFFs additionally report "
                    "ground area in m&sup2;."),
        "mode_example": "Try an example",
        "mode_own": "Upload my own images",
        "spinner": "Analyzing structural change...",
        "file_types": ["png", "jpg", "jpeg", "tif", "tiff"],
        "upload_before": "Drag and drop the earlier image",
        "upload_after": "Drag and drop the later image",
        "upload_label_before": "Before image",
        "upload_label_after": "After image",
        "upload_hint": "Upload both images to continue.",
    },
    "environment": {
        "card_finds": "Forest lost between two Sentinel-2 acquisitions.",
        # The second sentence is the point: it is what stops someone bringing
        # an ordinary photograph to this monitor.
        "card_needs": ("Two Sentinel-2 images of the same place, each with all "
                       "six bands: {bands}. Ordinary RGB photos will not work."),
        "card_extra": "",
        "purpose": "Find forest lost between two Sentinel-2 acquisitions.",
        "requires": ("Two Sentinel-2 images of the same place, each with all "
                     "six bands: {bands}."),
        # Plain language, on purpose: why an ordinary photo cannot work. No
        # architecture, no wavelengths, no channel talk.
        "requires_note": ("Forest loss is detected using infrared light that "
                          "ordinary RGB photos do not record."),
        "details": ("One array per date, covering the same area at the same "
                    "size. Previews shown here are true-colour composites "
                    "rendered from the supplied bands."),
        "mode_example": "Try an example",
        "mode_own": "Use my own data",
        "spinner": "Analyzing forest loss...",
        "file_types": ["npy"],
        "upload_before": "Drag and drop the earlier acquisition (.npy)",
        "upload_after": "Drag and drop the later acquisition (.npy)",
        "upload_label_before": "Before acquisition",
        "upload_label_after": "After acquisition",
        "upload_hint": "Upload both acquisitions to continue.",
    },
}

#: Used for a registered domain with no profile yet. Deliberately says nothing
#: specific rather than guessing at an input contract.
FALLBACK = {
    "card_finds": "Change between two views of the same place.",
    "card_needs": "Two inputs covering the same place at the same size.",
    "card_extra": "",
    "purpose": "Compare two views of the same place at different times.",
    "requires": "Two inputs covering the same place at the same size.",
    "requires_note": "",
    "details": "This engine declares its own input requirements.",
    "mode_example": "Try an example",
    "mode_own": "Use my own data",
    "spinner": "Analyzing change...",
    "file_types": None,                 # Streamlit: accept anything
    "upload_before": "Drag and drop the earlier input",
    "upload_after": "Drag and drop the later input",
    "upload_label_before": "Before",
    "upload_label_after": "After",
    "upload_hint": "Upload both inputs to continue.",
}

#: Domains with no engine, shown as unavailable. Any name here that DOES gain a
#: registry entry is dropped automatically by home.py, so this list can never
#: contradict the registry.
PLANNED = [
    ("disaster", "Disaster Monitor",
     "Flood, burn scar and post-disaster damage. Not implemented."),
]


def profile(domain):
    """Presentation profile for `domain`, never raising for an unknown name."""
    return PROFILES.get(domain, FALLBACK)


def _with_bands(template, bands):
    """Fill a {bands} placeholder, or leave the sentence alone without them."""
    bands = tuple(bands or ())
    return template.replace("{bands}", ", ".join(bands)) if bands else template


def requirement_text(domain, input_spec=None):
    """The "Requires" sentence, with band names taken from the engine itself.

    The copy lives here; the band list comes from the InputSpec the engine
    declares, so a model whose bands changed could not leave a stale promise on
    screen. A spec without band names simply leaves the template unfilled.
    """
    return _with_bands(profile(domain)["requires"],
                       getattr(input_spec, "bands", None))


def card_needs_text(domain, bands=()):
    """The Home card's "Needs" line, filled from the model card's declared bands.

    Home must not construct an engine just to draw a card, so the band list
    arrives from services.domain_card(), which reads the domain's model card.
    Same copy rule as requirement_text(): no band name is written here.
    """
    return _with_bands(profile(domain)["card_needs"], bands)
