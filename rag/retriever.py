"""
RAG retrieval — keyword-based matching against curated market sentences.
No sentence-transformers or FAISS loaded at runtime; saves ~100 MB RAM.
"""

from rag.setup import SAMPLE_SENTENCES

_CITY_ALIASES = {
    "delhi": ["delhi", "ncr", "dwarka", "rohini", "janakpuri", "saket", "lajpat"],
    "mumbai": ["mumbai", "bandra", "powai", "andheri", "thane", "navi mumbai",
               "borivali", "kandivali", "malad", "goregaon", "worli", "dadar",
               "chembur", "kurla", "mulund", "santacruz", "juhu", "lower parel"],
    "gurgaon": ["gurgaon", "gurugram", "dlf", "sohna", "manesar", "cyber city",
                "golf course", "palam vihar", "nirvana"],
    "hyderabad": ["hyderabad", "hitech city", "kondapur", "gachibowli", "jubilee",
                  "banjara", "madhapur", "kukatpally", "secunderabad", "manikonda",
                  "nallagandla", "financial district"],
    "bangalore": ["bangalore", "bengaluru", "whitefield", "indiranagar", "koramangala",
                  "electronic city", "hsr", "sarjapur", "marathahalli", "hebbal",
                  "yelahanka", "hennur", "jp nagar", "bannerghatta"],
    "kolkata": ["kolkata", "calcutta", "ballygunge", "salt lake", "rajarhat",
                "alipore", "new town", "howrah", "behala", "dum dum"],
    "pune": ["pune", "koregaon park", "wakad", "hinjewadi", "baner", "kothrud",
             "viman nagar", "hadapsar", "aundh", "magarpatta"],
    "chennai": ["chennai", "madras", "anna nagar", "omr", "adyar", "velachery", "porur"],
}

_PROP_KEYWORDS = {
    "apartment": ["apartment", "flat", "bhk"],
    "villa": ["villa", "independent house"],
    "penthouse": ["penthouse"],
    "studio": ["studio"],
}


def _city_score(sentence: str, city: str) -> int:
    s = sentence.lower()
    city_l = city.lower()
    aliases = _CITY_ALIASES.get(city_l, [city_l])
    return sum(1 for a in aliases if a in s)


def _prop_score(sentence: str, property_type: str) -> int:
    s = sentence.lower()
    pt = property_type.lower()
    keywords = _PROP_KEYWORDS.get(pt, [pt])
    return sum(1 for k in keywords if k in s)


def get_market_context(city: str, property_type: str, bedrooms: int) -> str:
    """
    Return top-4 market context sentences for the given property query.
    Uses keyword matching — no ML models loaded at runtime.
    """
    city = (city or "").strip()
    scored = []
    for s in SAMPLE_SENTENCES:
        score = _city_score(s, city) * 3 + _prop_score(s, property_type)
        if f"{bedrooms}BHK" in s or f"{bedrooms} BHK" in s:
            score += 2
        scored.append((score, s))

    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:4] if _ > 0]

    # Pad with generic trend sentences if not enough city-specific results
    if len(top) < 3:
        trend = [s for s in SAMPLE_SENTENCES if "2023" in s and city.lower() not in s.lower()]
        top += trend[:4 - len(top)]

    return "\n".join(f"• {s}" for s in top[:4])
