import json
import re

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "data/vector_db/bis.index"
METADATA_FILE = "data/vector_db/metadata.json"

SOURCE_BASE_URL = "https://www.bis.gov.in/"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    print("Loading BIS database...")

    index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    return index, metadata, model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    # Remove OCR Hindi section when English text exists
    hindi_markers = [
        "संपीडक",
        "यह मानक",
        "इस मानक",
        "आवश्यकताएँ",
        "प्रणाली",
        "सुरक्षाा"
    ]

    positions = []

    for marker in hindi_markers:

        position = text.find(marker)

        if position != -1:
            positions.append(position)

    if positions:

        first_hindi = min(positions)

        if first_hindi > 300:
            text = text[:first_hindi]

    return text.strip()


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def extract_sentences(text):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 15
    ]


# ============================================================
# QUERY KEYWORDS
# ============================================================

def get_query_keywords(query):

    stopwords = {
        "i", "want", "to", "use", "for", "the",
        "a", "an", "of", "and", "in", "on",
        "is", "are", "what", "which", "how",
        "can", "should", "with", "this",
        "that", "my", "from", "about",
        "requirements", "requirement"
    }

    words = re.findall(
        r"[a-zA-Z0-9]+",
        query.lower()
    )

    return {
        word
        for word in words
        if len(word) >= 3 and word not in stopwords
    }


# ============================================================
# RETRIEVAL
# ============================================================

def get_relevant_results(query, top_k=8):

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

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        result = metadata[index_id].copy()

        result["similarity"] = float(score)

        results.append(result)

    return results


# ============================================================
# STANDARD GROUPING
# ============================================================

def group_by_standard(results):

    grouped = {}

    for result in results:

        standard = (
            result.get("standard_number")
            or result.get("source_file")
            or "Unknown"
        )

        if standard not in grouped:

            grouped[standard] = {
                "standard_number": standard,
                "title": result.get("title", ""),
                "similarity": result["similarity"],
                "chunks": []
            }

        grouped[standard]["chunks"].append(result)

        grouped[standard]["similarity"] = max(
            grouped[standard]["similarity"],
            result["similarity"]
        )

    return sorted(
        grouped.values(),
        key=lambda x: x["similarity"],
        reverse=True
    )


# ============================================================
# RELEVANCE
# ============================================================

def calculate_keyword_overlap(query, text):

    query_keywords = get_query_keywords(query)

    if not query_keywords:
        return 0

    text_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            text.lower()
        )
    )

    overlap = query_keywords.intersection(text_words)

    return len(overlap) / len(query_keywords)


def calculate_group_relevance(query, group):

    all_text = " ".join(
        clean_text(chunk.get("text", ""))
        for chunk in group["chunks"]
    )

    keyword_score = calculate_keyword_overlap(
        query,
        all_text
    )

    semantic_score = group["similarity"]

    relevance = (
        semantic_score * 0.70
        +
        keyword_score * 0.30
    )

    return relevance


# ============================================================
# SENTENCE SELECTION
# ============================================================

def select_sentences(
    sentences,
    keywords,
    limit=2,
    max_chars=500
):

    selected = []
    seen = set()

    # First prefer sentences containing keywords
    for sentence in sentences:

        sentence_clean = sentence.strip()

        if not sentence_clean:
            continue

        sentence_lower = sentence_clean.lower()

        if any(
            keyword in sentence_lower
            for keyword in keywords
        ):

            normalized = sentence_lower

            if normalized not in seen:

                selected.append(sentence_clean)
                seen.add(normalized)

        if len(selected) >= limit:
            break

    # If no keyword matches, use first sentences
    if not selected:

        for sentence in sentences:

            sentence_clean = sentence.strip()

            if sentence_clean:

                selected.append(sentence_clean)

            if len(selected) >= limit:
                break

    # Limit total text length
    result = " ".join(selected)

    if len(result) > max_chars:

        result = result[:max_chars].rsplit(" ", 1)[0] + "..."

    return result


# ============================================================
# PRIMARY STANDARD
# ============================================================

def determine_primary_standard(
    query,
    groups
):

    if not groups:
        return None

    scored_groups = []

    for group in groups:

        relevance = calculate_group_relevance(
            query,
            group
        )

        group["relevance"] = relevance

        scored_groups.append(group)

    scored_groups.sort(
        key=lambda x: x["relevance"],
        reverse=True
    )

    primary = scored_groups[0]

    if primary["similarity"] < 0.40:

        return None

    return primary


# ============================================================
# PRIMARY TEXT
# ============================================================

def get_primary_text(primary):

    if not primary:
        return ""

    chunks = sorted(
        primary["chunks"],
        key=lambda x: x.get("chunk_index", 0)
    )

    return clean_text(
        " ".join(
            chunk.get("text", "")
            for chunk in chunks
        )
    )


# ============================================================
# STRUCTURED RESPONSE
# ============================================================

def build_structured_response(
    query,
    results
):

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if not results:

        return {
            "Primary IS":
                "No relevant BIS standard found.",

            "Aligned IS":
                "No closely related IS identified.",

            "Safety measures":
                "Not available in retrieved evidence.",

            "Implementation Measures":
                "Not available in retrieved evidence.",

            "Why use this":
                "No sufficiently relevant BIS standard was retrieved.",

            "Technical Specifications":
                "Not available in retrieved evidence.",

            "URL of Source of this data":
                SOURCE_BASE_URL
        }

    # --------------------------------------------------------
    # GROUP RESULTS
    # --------------------------------------------------------

    groups = group_by_standard(results)

    # --------------------------------------------------------
    # FIND PRIMARY
    # --------------------------------------------------------

    primary = determine_primary_standard(
        query,
        groups
    )

    if not primary:

        return {
            "Primary IS":
                "No directly relevant BIS standard identified in the current dataset.",

            "Aligned IS":
                "No closely related IS identified.",

            "Safety measures":
                "No specific safety information retrieved.",

            "Implementation Measures":
                "Verify the product against the applicable BIS standard.",

            "Why use this":
                "The current BIS dataset does not contain sufficient evidence for this query.",

            "Technical Specifications":
                "Not available in retrieved evidence.",

            "URL of Source of this data":
                SOURCE_BASE_URL
        }

    # --------------------------------------------------------
    # PRIMARY INFORMATION
    # --------------------------------------------------------

    standard_number = primary["standard_number"]

    title = (
        primary.get("title")
        or ""
    ).strip()

    primary_text = get_primary_text(primary)

    sentences = extract_sentences(
        primary_text
    )

    # ========================================================
    # SAFETY
    # ========================================================

    safety_keywords = [
        "safety",
        "safe",
        "hazard",
        "accident",
        "risk",
        "protection",
        "protective",
        "danger",
        "injury",
        "fire"
    ]

    safety = select_sentences(
        sentences,
        safety_keywords,
        limit=2,
        max_chars=400
    )

    if not safety:

        safety = (
            "No specific safety measures were identified "
            "in the retrieved text."
        )

    # ========================================================
    # IMPLEMENTATION
    # ========================================================

    implementation_keywords = [
        "requirement",
        "design",
        "construction",
        "installation",
        "install",
        "test",
        "testing",
        "procedure",
        "measurement",
        "inspection",
        "verification",
        "maintenance",
        "shall"
    ]

    implementation = select_sentences(
        sentences,
        implementation_keywords,
        limit=2,
        max_chars=400
    )

    if not implementation:

        implementation = (
            "Specific implementation measures were not "
            "identified in the retrieved text."
        )

    # ========================================================
    # WHY USE THIS
    # ========================================================

    why_keywords = [
        "purpose",
        "objective",
        "help",
        "minimize",
        "provide",
        "facilitate",
        "specifies",
        "covers",
        "intended"
    ]

    why_use = select_sentences(
        sentences,
        why_keywords,
        limit=1,
        max_chars=300
    )

    if not why_use:

        why_use = " ".join(
            sentences[:1]
        )

    if not why_use:

        why_use = (
            "This standard provides applicable requirements "
            "for the product or system."
        )

    # ========================================================
    # TECHNICAL SPECIFICATIONS
    # ========================================================

    technical_keywords = [
        "specification",
        "technical",
        "performance",
        "requirement",
        "dimensions",
        "voltage",
        "current",
        "power",
        "temperature",
        "efficiency",
        "rating",
        "test",
        "measurement",
        "construction",
        "material"
    ]

    technical = select_sentences(
        sentences,
        technical_keywords,
        limit=2,
        max_chars=400
    )

    if not technical:

        technical = (
            "Specific technical specifications were not "
            "identified in the retrieved text."
        )

    # ========================================================
    # ALIGNED IS
    # ========================================================

    aligned = []

    for group in groups:

        if group is primary:
            continue

        if group.get("similarity", 0) >= 0.45:

            standard = group["standard_number"]

            if standard not in aligned:

                aligned.append(standard)

    if aligned:

        aligned_text = ", ".join(
            aligned[:3]
        )

    else:

        aligned_text = (
            "No closely related IS identified."
        )

    # ========================================================
    # SOURCE URL
    # ========================================================

    source_url = (
        primary["chunks"][0].get("download_url")
        or primary["chunks"][0].get("view_url")
        or SOURCE_BASE_URL
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "Primary IS":
            f"{standard_number} — {title}",

        "Aligned IS":
            aligned_text,

        "Safety measures":
            safety,

        "Implementation Measures":
            implementation,

        "Why use this":
            why_use,

        "Technical Specifications":
            technical,

        "URL of Source of this data":
            source_url
    }


# ============================================================
# PRINT RESPONSE
# ============================================================

def print_response(response):

    print()
    print("=" * 80)
    print("BIS STANDARD ANALYSIS")
    print("=" * 80)

    print()
    print("Primary IS:")
    print(response["Primary IS"])

    print()
    print("Aligned IS:")
    print(response["Aligned IS"])

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


# ============================================================
# SEARCH
# ============================================================

def search(query, top_k=8):

    results = get_relevant_results(
        query,
        top_k
    )

    response = build_structured_response(
        query,
        results
    )

    print_response(response)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your BIS question: "
    )

    search(query)