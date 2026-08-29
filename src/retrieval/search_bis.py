import json
import re

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/vector_db/bis.index"
METADATA_FILE = "data/vector_db/metadata.json"

SOURCE_BASE_URL = "https://www.bis.gov.in/"


def load_database():

    print("Loading BIS database...")

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return index, metadata, model


def clean_text(text):

    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove repeated Hindi section when English content exists
    hindi_markers = [
        "संपीडक",
        "यह मानक",
        "इस मानक",
        "आवश्यकताएँ",
        "प्रणाली"
    ]

    positions = [
        text.find(marker)
        for marker in hindi_markers
        if text.find(marker) != -1
    ]

    if positions:
        first_hindi = min(positions)

        # Keep English portion before Hindi
        if first_hindi > 300:
            text = text[:first_hindi]

    return text.strip()


def extract_sentences(text):

    text = clean_text(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 20
    ]


def get_relevant_results(query, top_k=5):

    index, metadata, model = load_database()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id < 0:
            continue

        result = metadata[index_id].copy()

        result["similarity"] = float(score)

        results.append(result)

    return results


def build_structured_response(query, results):

    if not results:
        return {
            "Primary IS": "No relevant BIS standard found.",
            "Alliened IS": "Not available",
            "Safety measures": "Not available",
            "Implementation Measures": "Not available",
            "Why use this": "No sufficiently relevant BIS document was retrieved.",
            "Technical Specifications": "Not available",
            "URL of Source of this data": "Not available"
        }

    # ---------------------------------------------------------
    # PRIMARY STANDARD
    # ---------------------------------------------------------

    primary = results[0]

    primary_standard = (
        primary.get("standard_number")
        or primary.get("source_file")
        or "Unknown"
    )

    primary_title = primary.get("title", "").strip()

    primary_text = clean_text(primary.get("text", ""))

    # ---------------------------------------------------------
    # COLLECT SUPPORTING INFORMATION
    # ---------------------------------------------------------

    all_text = []

    for result in results:

        text = clean_text(result.get("text", ""))

        if text:
            all_text.append(text)

    combined_text = " ".join(all_text)

    sentences = extract_sentences(combined_text)

    # ---------------------------------------------------------
    # SAFETY MEASURES
    # ---------------------------------------------------------

    safety_keywords = [
        "safety",
        "safe",
        "hazard",
        "accident",
        "protection",
        "risk",
        "danger",
        "protective"
    ]

    safety_sentences = [
        s for s in sentences
        if any(
            keyword in s.lower()
            for keyword in safety_keywords
        )
    ]

    if safety_sentences:
        safety = " ".join(safety_sentences[:4])
    else:
        safety = (
            "The retrieved BIS material does not provide "
            "specific safety measures in the available text."
        )

    # ---------------------------------------------------------
    # IMPLEMENTATION MEASURES
    # ---------------------------------------------------------

    implementation_keywords = [
        "requirement",
        "design",
        "construction",
        "test",
        "testing",
        "procedure",
        "measurement",
        "inspection",
        "verification"
    ]

    implementation_sentences = [
        s for s in sentences
        if any(
            keyword in s.lower()
            for keyword in implementation_keywords
        )
    ]

    if implementation_sentences:
        implementation = " ".join(
            implementation_sentences[:5]
        )
    else:
        implementation = (
            "Implementation details are not explicitly "
            "provided in the retrieved text."
        )

    # ---------------------------------------------------------
    # WHY USE THIS
    # ---------------------------------------------------------

    why_keywords = [
        "purpose",
        "objective",
        "help",
        "minimize",
        "provide",
        "facilitate",
        "specifies",
        "covers"
    ]

    why_sentences = [
        s for s in sentences
        if any(
            keyword in s.lower()
            for keyword in why_keywords
        )
    ]

    if why_sentences:
        why_use = " ".join(why_sentences[:3])
    else:
        why_use = (
            primary_text[:800]
            if primary_text
            else "Purpose information is not available."
        )

    # ---------------------------------------------------------
    # TECHNICAL SPECIFICATIONS
    # ---------------------------------------------------------

    technical_keywords = [
        "specification",
        "technical",
        "performance",
        "requirements",
        "test",
        "standard",
        "design",
        "construction",
        "efficiency",
        "measurement"
    ]

    technical_sentences = [
        s for s in sentences
        if any(
            keyword in s.lower()
            for keyword in technical_keywords
        )
    ]

    if technical_sentences:
        technical = " ".join(
            technical_sentences[:6]
        )
    else:
        technical = (
            "Specific technical specifications were "
            "not available in the retrieved text."
        )

    # ---------------------------------------------------------
    # ALIGNED / RELATED IS
    # ---------------------------------------------------------

    aligned = []

    for result in results[1:]:

        standard = result.get("standard_number")

        if standard and standard != primary_standard:
            aligned.append(standard)

    if aligned:
        aligned_text = ", ".join(aligned)
    else:
        aligned_text = "No closely related IS identified."

    # ---------------------------------------------------------
    # SOURCE URL
    # ---------------------------------------------------------

    source_url = (
        primary.get("download_url")
        or primary.get("view_url")
        or SOURCE_BASE_URL
    )

    # ---------------------------------------------------------
    # RETURN STRUCTURED RESPONSE
    # ---------------------------------------------------------

    return {
        "Primary IS": (
            f"{primary_standard} — {primary_title}"
        ),

        "Alliened IS": aligned_text,

        "Safety measures": safety,

        "Implementation Measures": implementation,

        "Why use this": why_use,

        "Technical Specifications": technical,

        "URL of Source of this data": source_url
    }


def print_response(response):

    print()
    print("=" * 80)
    print("BIS STANDARD ANALYSIS")
    print("=" * 80)

    print()
    print("Primary IS:")
    print(response["Primary IS"])

    print()
    print("Alliened IS:")
    print(response["Alliened IS"])

    print()
    print("Safety measures:")
    print(response["Safety measures"])

    print()
    print("Implementation Measures:")
    print(response["Implementation Measures"])

    print()
    print("Why use this:")
    print(response["Why use this"])

    print()
    print("Technical Specifications:")
    print(response["Technical Specifications"])

    print()
    print("URL of Source of this data:")
    print(response["URL of Source of this data"])

    print()
    print("=" * 80)


def search(query, top_k=5):

    results = get_relevant_results(
        query,
        top_k
    )

    response = build_structured_response(
        query,
        results
    )

    print_response(response)


if __name__ == "__main__":

    query = input("\nEnter your BIS question: ")

    search(query)