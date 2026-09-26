"""Canonical cultural-group naming.

CultureBank stores cultural groups as free text ("American", "Americans",
"Swedish people"), while CulturalBench asks about countries ("United States",
"Japan", "Netherlands"). In the broken run this produced 2,410 distinct culture
strings for 22,990 records - "American" (3,651) and "Americans" (2,459) were
treated as different cultures - and there was **zero** exact overlap between the
45 benchmark culture labels and the 41 pathway culture labels.

That fragmentation is what starved pathway construction (which requires every
hop to share a culture) and broke culture-matched retrieval.

This module maps every surface form to one canonical key so entity resolution,
pathway construction, and retrieval all agree on what a culture is.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .utils import normalize_text

# Canonical key -> the surface forms that denote it (country names, demonyms,
# and common variants). Keys are lowercase and stable; they are what the rest of
# the pipeline compares on.
_CULTURE_FORMS: dict[str, tuple[str, ...]] = {
    "united states": ("usa", "us", "u.s.", "u.s.a.", "america", "american",
                      "americans", "united states of america", "yankee"),
    "united kingdom": ("uk", "u.k.", "britain", "great britain", "british",
                       "briton", "britons", "english", "england"),
    "scotland": ("scottish", "scots", "scot"),
    "wales": ("welsh",),
    "ireland": ("irish",),
    "canada": ("canadian", "canadians"),
    "australia": ("australian", "australians", "aussie", "aussies"),
    "new zealand": ("new zealander", "new zealanders", "kiwi", "kiwis"),
    "japan": ("japanese",),
    "china": ("chinese",),
    "hong kong": ("hongkong", "hong konger", "hong kongers"),
    "taiwan": ("taiwanese",),
    "korea": ("korean", "koreans", "south korea", "south korean",
              "south koreans", "north korea", "north korean"),
    "india": ("indian", "indians"),
    "pakistan": ("pakistani", "pakistanis"),
    "bangladesh": ("bangladeshi", "bangladeshis"),
    "nepal": ("nepali", "nepalese", "nepalis"),
    "sri lanka": ("sri lankan", "sri lankans"),
    "indonesia": ("indonesian", "indonesians"),
    "malaysia": ("malaysian", "malaysians"),
    "singapore": ("singaporean", "singaporeans"),
    "philippines": ("filipino", "filipinos", "filipina", "pinoy", "philippine"),
    "vietnam": ("vietnamese",),
    "thailand": ("thai", "thais"),
    "cambodia": ("cambodian", "cambodians", "khmer"),
    "myanmar": ("burmese", "burma"),
    "germany": ("german", "germans"),
    "france": ("french", "frenchman", "frenchmen"),
    "italy": ("italian", "italians"),
    "spain": ("spanish", "spaniard", "spaniards"),
    "portugal": ("portuguese",),
    "netherlands": ("dutch", "holland", "netherlander", "netherlanders"),
    "belgium": ("belgian", "belgians"),
    "switzerland": ("swiss",),
    "austria": ("austrian", "austrians"),
    "sweden": ("swedish", "swede", "swedes"),
    "norway": ("norwegian", "norwegians"),
    "denmark": ("danish", "dane", "danes"),
    "finland": ("finnish", "finn", "finns"),
    "iceland": ("icelandic", "icelander", "icelanders"),
    "poland": ("polish", "pole", "poles"),
    "czech republic": ("czech", "czechs", "czechia"),
    "slovakia": ("slovak", "slovaks", "slovakian"),
    "hungary": ("hungarian", "hungarians", "magyar"),
    "romania": ("romanian", "romanians"),
    "bulgaria": ("bulgarian", "bulgarians"),
    "greece": ("greek", "greeks"),
    "croatia": ("croatian", "croatians", "croat"),
    "serbia": ("serbian", "serbians", "serb"),
    "ukraine": ("ukrainian", "ukrainians"),
    "russia": ("russian", "russians"),
    "georgia": ("georgian", "georgians"),
    "armenia": ("armenian", "armenians"),
    "turkey": ("turkish", "turk", "turks", "turkiye"),
    "iran": ("iranian", "iranians", "persian", "persians"),
    "iraq": ("iraqi", "iraqis"),
    "israel": ("israeli", "israelis"),
    "palestine": ("palestinian", "palestinians"),
    "lebanon": ("lebanese",),
    "syria": ("syrian", "syrians"),
    "jordan": ("jordanian", "jordanians"),
    "saudi arabia": ("saudi", "saudis", "saudi arabian"),
    "united arab emirates": ("uae", "emirati", "emiratis"),
    "qatar": ("qatari", "qataris"),
    "kuwait": ("kuwaiti", "kuwaitis"),
    "egypt": ("egyptian", "egyptians"),
    "morocco": ("moroccan", "moroccans"),
    "algeria": ("algerian", "algerians"),
    "tunisia": ("tunisian", "tunisians"),
    "nigeria": ("nigerian", "nigerians"),
    "ghana": ("ghanaian", "ghanaians"),
    "kenya": ("kenyan", "kenyans"),
    "ethiopia": ("ethiopian", "ethiopians"),
    "south africa": ("south african", "south africans"),
    "zimbabwe": ("zimbabwean", "zimbabweans"),
    "uganda": ("ugandan", "ugandans"),
    "tanzania": ("tanzanian", "tanzanians"),
    "mexico": ("mexican", "mexicans"),
    "brazil": ("brazilian", "brazilians"),
    "argentina": ("argentinian", "argentinians", "argentine", "argentines"),
    "chile": ("chilean", "chileans"),
    "peru": ("peruvian", "peruvians"),
    "colombia": ("colombian", "colombians"),
    "venezuela": ("venezuelan", "venezuelans"),
    "ecuador": ("ecuadorian", "ecuadorians"),
    "bolivia": ("bolivian", "bolivians"),
    "uruguay": ("uruguayan", "uruguayans"),
    "cuba": ("cuban", "cubans"),
    "puerto rico": ("puerto rican", "puerto ricans"),
    "dominican republic": ("dominican", "dominicans"),
    "jamaica": ("jamaican", "jamaicans"),
    "haiti": ("haitian", "haitians"),
}

# Supra-national / regional groupings. Kept distinct from countries: they are
# valid cultural groups in CultureBank but must not be conflated with a member
# country (matching "European" to "France" would be wrong).
_REGION_FORMS: dict[str, tuple[str, ...]] = {
    "europe": ("european", "europeans"),
    "asia": ("asian", "asians"),
    "africa": ("african", "africans"),
    "latin america": ("latin american", "latin americans", "latino", "latinos",
                      "hispanic", "hispanics"),
    "middle east": ("middle eastern", "middle easterners", "arab", "arabs",
                    "arabic"),
    "scandinavia": ("scandinavian", "scandinavians", "nordic", "nordics"),
    "caribbean": ("caribbean islander",),
    "western": ("westerner", "westerners", "the west"),
}

# Descriptive wrappers that carry no cultural information.
_STRIP_SUFFIXES = (
    "people", "person", "culture", "cultures", "cultural", "society",
    "societies", "community", "communities", "population", "citizens",
    "citizen", "nationals", "context", "tradition", "traditions", "customs",
    "descent", "origin", "heritage", "speakers", "men", "women", "folks",
)

_STRIP_PREFIXES = ("the", "a", "an", "some", "most", "many", "all", "native")


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, forms in _CULTURE_FORMS.items():
        lookup[canonical] = canonical
        for form in forms:
            lookup[form] = canonical
    for canonical, forms in _REGION_FORMS.items():
        lookup[canonical] = canonical
        for form in forms:
            lookup[form] = canonical
    return lookup


_LOOKUP = _build_lookup()

_PUNCT = re.compile(r"[^\w\s-]")
_WS = re.compile(r"\s+")


def _strip_wrappers(text: str) -> str:
    """Remove leading articles and trailing descriptive nouns."""
    tokens = [t for t in text.split() if t]
    changed = True
    while changed and tokens:
        changed = False
        while tokens and tokens[0] in _STRIP_PREFIXES:
            tokens.pop(0)
            changed = True
        while tokens and tokens[-1] in _STRIP_SUFFIXES:
            tokens.pop()
            changed = True
    return " ".join(tokens)


def _depluralize(text: str) -> str:
    if len(text) > 3 and text.endswith("s") and not text.endswith("ss"):
        return text[:-1]
    return text


def canonical_culture(value: Optional[str]) -> str:
    """Return the canonical key for a cultural-group string ('' if unknown).

    Handles country/demonym pairs, plurals, and descriptive wrappers::

        canonical_culture("Americans")      -> "united states"
        canonical_culture("Swedish people") -> "sweden"
        canonical_culture("Japan")          -> "japan"
        canonical_culture("Japanese")       -> "japan"
    """
    text = normalize_text(value).lower()
    if not text:
        return ""
    text = _PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    if not text:
        return ""

    for candidate in (text, _strip_wrappers(text)):
        if not candidate:
            continue
        if candidate in _LOOKUP:
            return _LOOKUP[candidate]
        singular = _depluralize(candidate)
        if singular in _LOOKUP:
            return _LOOKUP[singular]

    # Unrecognized: return the cleaned, de-wrappered form so distinct free-text
    # groups still collapse together ("Swedish people" / "Swedish persons"),
    # rather than silently becoming "unknown".
    stripped = _strip_wrappers(text)
    return stripped or text


def cultures_match(a: Optional[str], b: Optional[str]) -> bool:
    """True when two culture strings denote the same cultural group."""
    ca, cb = canonical_culture(a), canonical_culture(b)
    if not ca or not cb:
        return False
    return ca == cb


def is_region(value: Optional[str]) -> bool:
    """True when the value denotes a supra-national region rather than a country."""
    return canonical_culture(value) in _REGION_FORMS


def canonical_cultures(values: Iterable[Optional[str]]) -> set[str]:
    """Canonicalize an iterable of culture strings, dropping empties."""
    out = {canonical_culture(v) for v in values}
    out.discard("")
    return out
