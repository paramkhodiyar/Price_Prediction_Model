import os
from pathlib import Path
from rag.setup import SAMPLE_SENTENCES, RAG_STORE_PATH, EMBEDDING_MODEL

# Global cache for the vector store to avoid reloading on every request
_VECTORSTORE = None

def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        if os.path.exists(RAG_STORE_PATH):
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            _VECTORSTORE = FAISS.load_local(RAG_STORE_PATH, embedder, allow_dangerous_deserialization=True)
    return _VECTORSTORE

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

def _keyword_search_fallback(city: str, property_type: str, bedrooms: int) -> str:
    """Fallback keyword matching if FAISS is not available."""
    city = (city or "").lower().strip()
    pt = (property_type or "").lower().strip()
    
    scored = []
    for s in SAMPLE_SENTENCES:
        score = 0
        s_lower = s.lower()
        
        # City/Alias match
        aliases = _CITY_ALIASES.get(city, [city])
        if any(a in s_lower for a in aliases):
            score += 3
            
        # Property type match
        if pt in s_lower:
            score += 1
            
        # Bedrooms match
        if f"{bedrooms}bhk" in s_lower.replace(" ", ""):
            score += 2
            
        if score > 0:
            scored.append((score, s))
            
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:4]]
    return "\n".join(f"• {s}" for s in top)

def get_market_context(city: str, property_type: str, bedrooms: int) -> str:
    """
    Return market context sentences using FAISS vector similarity search.
    Falls back to keyword matching if the vector store is not found.
    """
    query = f"Real estate market trends for {bedrooms}BHK {property_type} in {city} in 2023"
    
    vs = _get_vectorstore()
    if vs:
        try:
            # Search for similar market data points
            docs = vs.similarity_search(query, k=4)
            if docs:
                return "\n".join(f"• {doc.page_content}" for doc in docs)
        except Exception:
            pass
            
    # Fallback if FAISS fails or store is missing
    return _keyword_search_fallback(city, property_type, bedrooms)
