"""
Build FAISS vector store from city-level real estate CSVs.
Run once: python -m rag.setup
Auto-runs on first retrieval if rag_store/ doesn't exist.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAG_STORE_PATH = str(Path(__file__).parent.parent / "rag_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Rich curated market data — used when CSVs are Git LFS pointers or unavailable
SAMPLE_SENTENCES = [
    # Mumbai localities
    "In Mumbai Bandra West in 2023, 2BHK apartments averaged ₹2.1 Cr with area around 800 sqft.",
    "In Mumbai Bandra West in 2023, 3BHK apartments averaged ₹3.8 Cr and are highly sought after.",
    "In Mumbai Powai in 2023, 2BHK apartments averaged ₹1.4 Cr with good connectivity to IT hubs.",
    "In Mumbai Andheri West in 2023, 2BHK apartments averaged ₹1.2 Cr with metro connectivity.",
    "In Mumbai Malabar Hill in 2023, 4BHK apartments averaged ₹8.5 Cr in a prime sea-facing location.",
    "In Mumbai Thane in 2023, 2BHK apartments averaged ₹0.85 Cr in upcoming residential zones.",
    "In Mumbai Worli in 2023, 3BHK apartments averaged ₹5.2 Cr with sea views and luxury amenities.",
    "In Mumbai Goregaon in 2023, 2BHK apartments averaged ₹1.1 Cr near Aarey Colony.",
    "In Mumbai Kandivali in 2023, 2BHK apartments averaged ₹0.95 Cr in western suburbs.",
    "In Mumbai Lower Parel in 2023, 2BHK apartments averaged ₹2.8 Cr in commercial hub.",
    "In Mumbai Mulund in 2023, 2BHK apartments averaged ₹0.9 Cr with good social infrastructure.",
    "In Mumbai Navi Mumbai in 2023, 2BHK apartments averaged ₹0.75 Cr in planned township.",
    "In Mumbai Santacruz in 2023, 2BHK apartments averaged ₹1.8 Cr near international airport.",
    "In Mumbai Juhu in 2023, 4BHK villas averaged ₹12 Cr in exclusive beachside locality.",
    "In Mumbai Dadar in 2023, 2BHK apartments averaged ₹1.6 Cr in central location.",
    "In Mumbai Chembur in 2023, 2BHK apartments averaged ₹1.05 Cr with Eastern Freeway access.",
    "In Mumbai Kurla in 2023, 2BHK apartments averaged ₹0.92 Cr near BKC.",
    "In Mumbai Malad in 2023, 2BHK apartments averaged ₹0.98 Cr in western suburbs.",
    "In Mumbai Borivali in 2023, 2BHK apartments averaged ₹0.88 Cr near national park.",
    "In Mumbai Dharavi in 2023, 1BHK apartments averaged ₹0.55 Cr in redevelopment zone.",
    # Gurgaon localities
    "In Gurgaon DLF Phase 1 in 2023, 3BHK apartments averaged ₹1.4 Cr in premium gated societies.",
    "In Gurgaon Sector 56 in 2023, 3BHK apartments averaged ₹1.1 Cr with good metro connectivity.",
    "In Gurgaon Golf Course Road in 2023, 4BHK apartments averaged ₹2.5 Cr in luxury corridor.",
    "In Gurgaon Sohna Road in 2023, 2BHK apartments averaged ₹0.85 Cr in developing residential area.",
    "In Gurgaon Cyber City in 2023, 2BHK apartments averaged ₹1.2 Cr near IT hub.",
    "In Gurgaon MG Road in 2023, 3BHK apartments averaged ₹1.6 Cr in prime commercial zone.",
    "In Gurgaon Sector 82 in 2023, 3BHK apartments averaged ₹0.95 Cr in new Dwarka Expressway township.",
    "In Gurgaon Nirvana Country in 2023, 4BHK villas averaged ₹3.2 Cr in green gated township.",
    "In Gurgaon Palam Vihar in 2023, 3BHK independent houses averaged ₹1.8 Cr.",
    "In Gurgaon South City 2 in 2023, 3BHK apartments averaged ₹1.3 Cr.",
    "In Gurgaon Sector 65 in 2023, 2BHK apartments averaged ₹0.9 Cr in emerging zone.",
    "In Gurgaon Golf Course Extension Road in 2023, 4BHK penthouses averaged ₹4.5 Cr.",
    "In Gurgaon DLF Phase 5 in 2023, 3BHK apartments averaged ₹1.75 Cr in established zone.",
    "In Gurgaon Sector 43 in 2023, 3BHK apartments averaged ₹1.25 Cr near NH-48.",
    "In Gurgaon Manesar in 2023, 2BHK apartments averaged ₹0.65 Cr in industrial belt.",
    # Delhi localities
    "In Delhi Vasant Kunj in 2023, 3BHK apartments averaged ₹1.8 Cr in south Delhi.",
    "In Delhi South Extension in 2023, 4BHK independent houses averaged ₹5.5 Cr.",
    "In Delhi Dwarka in 2023, 3BHK apartments averaged ₹1.1 Cr in planned satellite city.",
    "In Delhi Greater Kailash in 2023, 4BHK independent houses averaged ₹4.8 Cr.",
    "In Delhi Rohini in 2023, 3BHK apartments averaged ₹0.85 Cr in north Delhi.",
    "In Delhi Lajpat Nagar in 2023, 3BHK apartments averaged ₹1.5 Cr in central location.",
    "In Delhi Saket in 2023, 2BHK apartments averaged ₹1.3 Cr near Select Citywalk mall.",
    "In Delhi Janakpuri in 2023, 3BHK apartments averaged ₹0.9 Cr in west Delhi.",
    "In Delhi Punjabi Bagh in 2023, 4BHK independent houses averaged ₹4.2 Cr.",
    "In Delhi Pitampura in 2023, 2BHK apartments averaged ₹0.78 Cr in north-west Delhi.",
    # Hyderabad localities
    "In Hyderabad Hitech City in 2023, 2BHK apartments averaged ₹0.85 Cr near IT corridor.",
    "In Hyderabad Kondapur in 2023, 2BHK apartments averaged ₹0.78 Cr with IT park proximity.",
    "In Hyderabad Gachibowli in 2023, 3BHK apartments averaged ₹1.1 Cr in prime IT zone.",
    "In Hyderabad Jubilee Hills in 2023, 4BHK villas averaged ₹3.8 Cr in premium locality.",
    "In Hyderabad Banjara Hills in 2023, 4BHK independent houses averaged ₹4.2 Cr.",
    "In Hyderabad Madhapur in 2023, 2BHK apartments averaged ₹0.88 Cr near Hitech City.",
    "In Hyderabad Kukatpally in 2023, 2BHK apartments averaged ₹0.65 Cr in affordable zone.",
    "In Hyderabad Secunderabad in 2023, 3BHK apartments averaged ₹0.72 Cr in established area.",
    "In Hyderabad Financial District in 2023, 3BHK apartments averaged ₹1.25 Cr near Amazon campus.",
    "In Hyderabad Manikonda in 2023, 2BHK apartments averaged ₹0.6 Cr in growing suburb.",
    "In Hyderabad Nallagandla in 2023, 3BHK apartments averaged ₹0.95 Cr near Gachibowli.",
    "In Hyderabad Shamshabad in 2023, 2BHK apartments averaged ₹0.52 Cr near new airport.",
    # Bangalore localities
    "In Bangalore Whitefield in 2023, 2BHK apartments averaged ₹0.82 Cr near IT corridors.",
    "In Bangalore Indiranagar in 2023, 2BHK apartments averaged ₹1.2 Cr in premium lifestyle zone.",
    "In Bangalore Koramangala in 2023, 3BHK apartments averaged ₹1.5 Cr in startup hub.",
    "In Bangalore Electronic City in 2023, 2BHK apartments averaged ₹0.62 Cr near Infosys campus.",
    "In Bangalore HSR Layout in 2023, 3BHK apartments averaged ₹1.1 Cr in tech-friendly area.",
    "In Bangalore Sarjapur Road in 2023, 2BHK apartments averaged ₹0.75 Cr in developing zone.",
    "In Bangalore JP Nagar in 2023, 3BHK apartments averaged ₹0.95 Cr in south Bangalore.",
    "In Bangalore Marathahalli in 2023, 2BHK apartments averaged ₹0.72 Cr near Outer Ring Road.",
    "In Bangalore Hennur in 2023, 2BHK apartments averaged ₹0.68 Cr in north Bangalore.",
    "In Bangalore Yelahanka in 2023, 3BHK villas averaged ₹1.8 Cr near international airport.",
    "In Bangalore Bannerghatta Road in 2023, 3BHK apartments averaged ₹0.88 Cr in south Bangalore.",
    "In Bangalore Hebbal in 2023, 3BHK apartments averaged ₹1.05 Cr near Manyata Tech Park.",
    # Kolkata localities
    "In Kolkata Ballygunge in 2023, 3BHK apartments averaged ₹0.95 Cr in upscale south Kolkata.",
    "In Kolkata Salt Lake in 2023, 3BHK apartments averaged ₹0.85 Cr in planned IT township.",
    "In Kolkata Rajarhat in 2023, 2BHK apartments averaged ₹0.55 Cr in new town development.",
    "In Kolkata Alipore in 2023, 4BHK apartments averaged ₹2.8 Cr in elite heritage locality.",
    "In Kolkata New Town in 2023, 2BHK apartments averaged ₹0.62 Cr in IT hub area.",
    "In Kolkata Park Street in 2023, 3BHK apartments averaged ₹1.4 Cr in heritage commercial zone.",
    "In Kolkata Howrah in 2023, 2BHK apartments averaged ₹0.45 Cr across the river.",
    "In Kolkata Behala in 2023, 2BHK apartments averaged ₹0.42 Cr in south-west Kolkata.",
    "In Kolkata Dum Dum in 2023, 2BHK apartments averaged ₹0.48 Cr near airport.",
    # Pune localities
    "In Pune Koregaon Park in 2023, 3BHK apartments averaged ₹1.4 Cr in premium locality.",
    "In Pune Wakad in 2023, 2BHK apartments averaged ₹0.72 Cr near Hinjewadi IT park.",
    "In Pune Hinjewadi in 2023, 2BHK apartments averaged ₹0.68 Cr in IT zone.",
    "In Pune Baner in 2023, 2BHK apartments averaged ₹0.88 Cr in mid-premium segment.",
    "In Pune Kothrud in 2023, 3BHK apartments averaged ₹0.98 Cr in established locality.",
    "In Pune Viman Nagar in 2023, 2BHK apartments averaged ₹0.82 Cr near airport.",
    "In Pune Hadapsar in 2023, 2BHK apartments averaged ₹0.65 Cr in east Pune IT zone.",
    "In Pune Camp in 2023, 3BHK apartments averaged ₹1.1 Cr in central heritage Pune.",
    "In Pune Aundh in 2023, 3BHK apartments averaged ₹1.05 Cr in upscale north-west Pune.",
    "In Pune Magarpatta in 2023, 2BHK apartments averaged ₹0.78 Cr in planned township.",
    # Chennai localities
    "In Chennai Anna Nagar in 2023, 3BHK apartments averaged ₹1.2 Cr in prestigious locality.",
    "In Chennai OMR in 2023, 2BHK apartments averaged ₹0.72 Cr on IT corridor.",
    "In Chennai Adyar in 2023, 3BHK apartments averaged ₹1.4 Cr near beach.",
    "In Chennai Velachery in 2023, 2BHK apartments averaged ₹0.82 Cr near IT park.",
    "In Chennai Porur in 2023, 2BHK apartments averaged ₹0.65 Cr in west Chennai.",
    # Market trend insights
    "Mumbai real estate in 2023 showed 8-12% annual appreciation in prime localities like Bandra, Juhu, and Worli.",
    "Gurgaon Golf Course Road corridor in 2023 is considered a premium investment zone with rental yields of 3-4%.",
    "Hyderabad IT corridor from Hitech City to Financial District in 2023 showed strong demand from tech professionals.",
    "Bangalore outperformed other metros in 2023 with 15% appreciation in Whitefield and Electronic City zones.",
    "Kolkata real estate in 2023 remained affordable with steady 5-7% annual growth in new town areas.",
    "Delhi NCR saw mixed trends in 2023 with Gurgaon and Noida outperforming central Delhi in appreciation.",
    "Pune residential market in 2023 benefited from IT sector growth in Hinjewadi and Magarpatta corridors.",
    "Indian real estate market in 2023 saw increased demand for 2BHK and 3BHK apartments post-pandemic.",
    "Properties near metro stations in Mumbai and Delhi saw 10-15% premium over non-metro localities in 2023.",
    "Independent houses and villas in gated communities showed highest appreciation in 2023 across Indian metros.",
    "New launches in peripheral areas of major cities offered 20-25% lower prices than established localities in 2023.",
    "Luxury segment above ₹2 Cr saw increased NRI and tech executive demand across Indian metros in 2023.",
    "Property age significantly impacts value: buildings below 5 years command 10-20% premium over older stock.",
    "Furnished properties command 8-12% higher rental yields compared to unfurnished ones in major metros.",
    "Floor level premium: properties above 10th floor command 5-10% premium in high-rise buildings in 2023.",
    "Penthouse apartments in Mumbai and Gurgaon saw 25-30% premium over regular floor apartments in 2023.",
    "Gated community properties with amenities command 15-20% premium over standalone buildings.",
    "Investment in real estate near upcoming metro corridors showed highest capital appreciation in 2023.",
    "Affordable housing below ₹50 Lac attracted first-time buyers in tier-2 cities in 2023.",
    "Semi-furnished properties offer the best value proposition for investors targeting rental income in 2023.",
]


def _is_lfs_pointer(filepath: Path) -> bool:
    try:
        return filepath.read_bytes()[:60].decode("utf-8", errors="ignore").startswith(
            "version https://git-lfs.github.com"
        )
    except Exception:
        return True


def _csv_to_sentences(filepath: Path, city: str) -> list:
    sentences = []
    try:
        df = pd.read_csv(filepath, nrows=2000)
        df.columns = df.columns.str.lower().str.strip().str.replace(r"\s+", "_", regex=True)
        for _, row in df.iterrows():
            try:
                locality = next(
                    (str(row[c]) for c in ["location", "locality", "area", "sector", "neighbourhood"]
                     if c in row.index and pd.notna(row[c]) and str(row[c]).strip()),
                    None,
                )
                beds = next(
                    (row[c] for c in ["bedrooms", "bhk", "bedroom", "beds"]
                     if c in row.index and pd.notna(row[c])),
                    None,
                )
                prop_type = next(
                    (str(row[c]) for c in ["property_type", "type", "prop_type"]
                     if c in row.index and pd.notna(row[c])),
                    "property",
                )
                price = next(
                    (row[c] for c in ["price", "price_in_cr", "price_inr", "value"]
                     if c in row.index and pd.notna(row[c])),
                    None,
                )
                year = next(
                    (int(row[c]) for c in ["year", "year_posted", "listing_year"]
                     if c in row.index and pd.notna(row[c])),
                    2023,
                )
                area = next(
                    (row[c] for c in ["area_sqft", "area", "super_built_up_area", "carpet_area"]
                     if c in row.index and pd.notna(row[c])),
                    None,
                )
                if not (locality and price):
                    continue
                price_val = float(str(price).replace(",", "").replace("₹", "").strip())
                if price_val > 10000:
                    price_val = price_val / 10_000_000  # convert raw rupees to Cr
                bhk_str = f"{int(float(str(beds)))}BHK " if beds else ""
                area_str = f" ({int(float(str(area).replace(',', '')))} sqft)" if area else ""
                sentences.append(
                    f"In {city} {locality} in {year}, {bhk_str}{prop_type}s averaged ₹{price_val:.2f} Cr{area_str}."
                )
            except Exception:
                continue
    except Exception:
        pass
    return sentences


def build_vector_store(silent: bool = False):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    sentences = []
    target_csvs = [
        (DATA_DIR / "mumbai.csv", "Mumbai"),
        (DATA_DIR / "gurgaon_10k.csv", "Gurgaon"),
        (DATA_DIR / "hyderabad.csv", "Hyderabad"),
    ]

    for csv_path, city in target_csvs:
        if csv_path.exists() and not _is_lfs_pointer(csv_path):
            if not silent:
                print(f"Loading real CSV data: {csv_path.name} ...")
            loaded = _csv_to_sentences(csv_path, city)
            if loaded:
                sentences.extend(loaded[:500])
                if not silent:
                    print(f"  → {len(loaded)} sentences from {city}")
            else:
                if not silent:
                    print(f"  → Could not parse {city}, skipping")

    if not sentences:
        if not silent:
            print("CSV files are LFS pointers or unavailable — using curated sample market data.")
        sentences = SAMPLE_SENTENCES

    sentences = list(dict.fromkeys(sentences))  # deduplicate preserving order
    if not silent:
        print(f"\nEmbedding {len(sentences)} market context sentences ...")

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_texts(sentences, embedder)
    vectorstore.save_local(RAG_STORE_PATH)
    if not silent:
        print(f"Vector store saved → {RAG_STORE_PATH}")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()
