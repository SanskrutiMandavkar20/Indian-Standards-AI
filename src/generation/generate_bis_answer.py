import re


SOURCE_BASE_URL = "https://www.bis.gov.in/"


def clean_text(text):
    """Clean extracted BIS text."""

    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    # Remove obvious Hindi continuation when English content exists
    hindi_markers = [
        "संपीडक",
        "यह मानक",
        "इस मानक",
        "आवश्यकताएँ",
        "प्रणाली",
        "भारत की राष्ट्रीय",
    ]

    positions = [
        text.find(marker)
        for marker in hindi_markers
        if text.find(marker) != -1
    ]

    if positions:
        first_hindi = min(positions)

        if first_hindi > 300:
            text = text[:first_hindi]

    return text.strip()


def extract_sentences(text):

    text = clean_text(text)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        s.strip()
        for s in sentences
        if len(s.strip()) > 25
    ]


def extract_standard_number(text, source_file=""):

    patterns = [
        r"\bIS\s*/?\s*IEC\s*[\w()/-]+\s*:\s*\d{4}",
        r"\bIS\s+\d+(?:\s*\([^)]*\))?\s*:\s*\d{4}",
        r"\bSP\s+\d+\s*:\s*\d{4}",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return re.sub(
                r"\s+",
                " ",
                match.group(0)
            ).strip()

    # Fallback to filename
    if source_file:

        name = source_file.rsplit(".", 1)[0]

        return name.replace("_", " ")

    return "Unknown"


def extract_title(text):

    text = clean_text(text)

    # Try to identify title from the first line/sentence
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if not lines:
        return ""

    first_line = lines[0]

    # Remove standard number from beginning
    first_line = re.sub(
        r"^(IS|SP)\s*[\w()/\-\s]*:\s*\d{4}\s*",
        "",
        first_line,
        flags=re.IGNORECASE
    )

    return first_line.strip()


def select_primary(results):

    if not results:
        return None

    # Only consider reasonably relevant results.
    relevant = [
        r for r in results
        if r.get("similarity", 0) >= 0.50
    ]

    if not relevant:
        return None

    return relevant[0]


def get_related_standards(primary, results):

    if not primary:
        return []

    primary_source = primary.get("source_file")

    related = []

    for result in results:

        if result.get("source_file") == primary_source:
            continue

        similarity = result.get("similarity", 0)

        # Only include reasonably related documents
        if similarity >= 0.40:

            text = clean_text(
                result.get("text", "")
            )

            standard = extract_standard_number(
                text,
                result.get("source_file", "")
            )

            if standard not in related:
                related.append(standard)

    return related


def extract_evidence(results, keywords, limit=4):

    evidence = []

    for result in results:

        text = clean_text(
            result.get("text", "")
        )

        sentences = extract_sentences(text)

        for sentence in sentences:

            sentence_lower = sentence.lower()

            if any(
                keyword in sentence_lower
                for keyword in keywords
            ):

                if sentence not in evidence:

                    evidence.append(sentence)

            if len(evidence) >= limit:
                return evidence

    return evidence


def build_structured_response(query, results):

    primary = select_primary(results)

    # ---------------------------------------------------------
    # NO SUFFICIENTLY RELEVANT STANDARD
    # ---------------------------------------------------------

    if not primary:

        return {
            "Primary IS":
                "No directly applicable BIS standard identified "
                "in the current dataset.",

            "Aligned IS":
                "No closely related IS identified.",

            "Safety measures":
                "The current retrieved evidence does not provide "
                "sufficient information for a specific safety "
                "recommendation.",

            "Implementation Measures":
                "Verify the product against the applicable BIS "
                "standard before implementation.",

            "Why use this":
                "The current BIS dataset does not contain "
                "sufficient evidence to identify a directly "
                "applicable standard for this query.",

            "Technical Specifications":
                "Not available in retrieved evidence.",

            "URL of Source of this data":
                SOURCE_BASE_URL
        }

    # ---------------------------------------------------------
    # PRIMARY IS
    # ---------------------------------------------------------

    primary_text = clean_text(
        primary.get("text", "")
    )

    standard_number = extract_standard_number(
        primary_text,
        primary.get("source_file", "")
    )

    title = extract_title(primary_text)

    primary_is = f"{standard_number} — {title}"

    # ---------------------------------------------------------
    # ALIGNED IS
    # ---------------------------------------------------------

    aligned = get_related_standards(
        primary,
        results
    )

    aligned_text = (
        ", ".join(aligned)
        if aligned
        else "No closely related IS identified."
    )

    # ---------------------------------------------------------
    # SAFETY
    # ---------------------------------------------------------

    safety = extract_evidence(
        results,
        [
            "safety",
            "safe",
            "hazard",
            "risk",
            "protection",
            "danger",
            "accident",
            "protective",
        ],
        limit=4
    )

    if safety:

        safety_text = " ".join(safety)

    else:

        safety_text = (
            "The retrieved BIS material does not provide "
            "specific safety measures in the available text."
        )

    # ---------------------------------------------------------
    # IMPLEMENTATION
    # ---------------------------------------------------------

    implementation = extract_evidence(
        results,
        [
            "requirement",
            "design",
            "installation",
            "construction",
            "test",
            "testing",
            "procedure",
            "inspection",
            "verification",
            "maintenance",
        ],
        limit=4
    )

    if implementation:

        implementation_text = " ".join(
            implementation
        )

    else:

        implementation_text = (
            "Implementation details are not explicitly "
            "provided in the retrieved text."
        )

    # ---------------------------------------------------------
    # WHY USE THIS
    # ---------------------------------------------------------

    why = extract_evidence(
        results,
        [
            "purpose",
            "objective",
            "provide",
            "help",
            "minimize",
            "facilitate",
            "specifies",
            "covers",
        ],
        limit=3
    )

    if why:

        why_text = " ".join(why)

    else:

        why_text = primary_text[:800]

    # ---------------------------------------------------------
    # TECHNICAL SPECIFICATIONS
    # ---------------------------------------------------------

    technical = extract_evidence(
        results,
        [
            "specification",
            "technical",
            "performance",
            "requirement",
            "standard",
            "design",
            "construction",
            "efficiency",
            "measurement",
            "test",
        ],
        limit=5
    )

    if technical:

        technical_text = " ".join(
            technical
        )

    else:

        technical_text = (
            "Specific technical specifications were not "
            "available in the retrieved evidence."
        )

    # ---------------------------------------------------------
    # SOURCE
    # ---------------------------------------------------------

    source_url = (
        primary.get("download_url")
        or primary.get("view_url")
        or SOURCE_BASE_URL
    )

    return {

        "Primary IS":
            primary_is,

        "Aligned IS":
            aligned_text,

        "Safety measures":
            safety_text,

        "Implementation Measures":
            implementation_text,

        "Why use this":
            why_text,

        "Technical Specifications":
            technical_text,

        "URL of Source of this data":
            source_url
    }


def print_response(response):

    print()
    print("=" * 80)
    print("BIS STANDARD ANALYSIS")
    print("=" * 80)

    print("\nPrimary IS:")
    print(response["Primary IS"])

    print("\nAligned IS:")
    print(response["Aligned IS"])

    print("\nSafety measures:")
    print(response["Safety measures"])

    print("\nImplementation Measures:")
    print(response["Implementation Measures"])

    print("\nWhy use this:")
    print(response["Why use this"])

    print("\nTechnical Specifications:")
    print(response["Technical Specifications"])

    print("\nURL of Source of this data:")
    print(response["URL of Source of this data"])

    print()
    print("=" * 80)


if __name__ == "__main__":

    print(
        "This module provides BIS answer generation."
    )

    print(
        "It is called by the retrieval pipeline."
    )