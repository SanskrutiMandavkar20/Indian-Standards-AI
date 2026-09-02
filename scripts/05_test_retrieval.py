import json
import re
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import pymupdf
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "vector_db"
STANDARD_PDF_DIR = BASE_DIR / "data" / "pdfs"
CERTIFICATION_PDF_DIR = BASE_DIR / "data" / "certification"

STANDARD_INDEX = VECTOR_DIR / "standards.index"
STANDARD_METADATA = VECTOR_DIR / "standards_metadata.json"

CERT_INDEX = VECTOR_DIR / "certification.index"
CERT_METADATA = VECTOR_DIR / "certification_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 20


# ============================================================
# GLOBAL REQUIREMENT
# ============================================================

CURRENT_REQUIREMENT = {}


# ============================================================
# USER INPUT
# ============================================================

def get_user_requirement():

    print("\nINDIAN STANDARDS RECOMMENDATION ENGINE")

    print("""
Enter the procurement requirement.

Product Title and Description are required.
Category, Quantity and Additional Requirements are optional.
""")

    while True:

        title = input(
            "Product Title:\n> "
        ).strip()

        if title:
            break

        print(
            "Product Title cannot be empty.\n"
        )

    while True:

        description = input(
            "\nDescription:\n> "
        ).strip()

        if description:
            break

        print(
            "Description cannot be empty.\n"
        )

    category = input(
        "\nCategory (optional):\n> "
    ).strip()

    quantity = input(
        "\nQuantity (optional):\n> "
    ).strip()

    details = input(
        "\nAdditional Requirements (optional):\n> "
    ).strip()

    return {
        "title": title,
        "description": description,
        "category": category,
        "quantity": quantity,
        "details": details,
    }


# ============================================================
# REQUIREMENT VERIFICATION
# ============================================================

def display_requirement_verification(requirement):

    print("\n")
    print(
        "STAGE 2 OF 8  •  REQUIREMENT VERIFICATION"
    )

    print(
        "\nWe understood your requirement as:"
    )

    print("\nPRODUCT")
    print(
        f"  {requirement['title']}"
    )

    if requirement["category"]:

        print("\nCATEGORY")
        print(
            f"  {requirement['category']}"
        )

    if requirement["quantity"]:

        print("\nQUANTITY")
        print(
            f"  {requirement['quantity']}"
        )

    print("\nINTENDED PURPOSE")
    print(
        f"  {requirement['description']}"
    )

    if requirement["details"]:

        print(
            "\nKEY TECHNICAL REQUIREMENTS "
            "DETECTED BY AI"
        )

        requirements = split_requirements(
            requirement["details"]
        )

        for item in requirements:

            print(
                f"  ✓ {item}"
            )

    print(
        "\nAnalyzing procurement requirement..."
    )


def split_requirements(text):

    if not text:
        return []

    parts = re.split(
        r"(?:\n|;|•|\u2022)",
        text
    )

    cleaned = []

    for part in parts:

        part = re.sub(
            r"^\s*[-*]\s*",
            "",
            part
        ).strip()

        if part:
            cleaned.append(part)

    if not cleaned:
        return [text.strip()]

    return cleaned


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^\w\s:./()-]",
        "",
        text
    )

    return text.strip()


def tokenize(text):

    return set(
        normalize_text(text).split()
    )


# ============================================================
# EXTRACT INDIAN STANDARD NUMBERS
# ============================================================

def extract_standard_numbers(text):

    if not text:
        return []

    pattern = (
        r"\bIS\s+\d{4,6}"
        r"(?:\s*\([^)]{1,100}\))?"
        r"\s*:\s*\d{4}\b"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE
    )

    results = []

    for match in matches:

        match = re.sub(
            r"\s+",
            " ",
            match.strip()
        )

        results.append(
            match.upper()
        )

    return list(
        dict.fromkeys(results)
    )


# ============================================================
# STANDARD TITLE EXTRACTION
# ============================================================

def extract_standard_title(
    evidence,
    standard_number
):

    if not evidence or not standard_number:
        return None

    # Example:
    # IS 17873:2022
    # IS 17873 : 2022

    number_match = re.search(
        r"IS\s+(\d{4,6})"
        r"(?:\s*\([^)]{1,100}\))?"
        r"\s*:\s*(\d{4})",
        standard_number,
        flags=re.IGNORECASE
    )

    if not number_match:
        return None

    standard_regex = (
        r"\bIS\s+"
        + re.escape(
            number_match.group(1)
        )
        + r"\s*:\s*"
        + re.escape(
            number_match.group(2)
        )
        + r"\b"
    )

    pattern = re.compile(
        standard_regex
        + r"\s+(.{5,250})",
        flags=re.IGNORECASE
    )

    for item in evidence:

        text = item["metadata"].get(
            "text",
            ""
        )

        match = pattern.search(text)

        if not match:
            continue

        title = match.group(1)

        # Stop at another IS number
        title = re.split(
            r"\s+IS\s+\d{4,6}"
            r"(?:\s*\([^)]{1,100}\))?"
            r"\s*:\s*\d{4}",
            title,
            flags=re.IGNORECASE
        )[0]

        # Stop at common descriptive sentence
        title = re.split(
            r"\bThis standard\b",
            title,
            flags=re.IGNORECASE
        )[0]

        title = title.strip(
            " :-–—."
        )

        if title:
            return title

    return None


# ============================================================
# JSON LOADER
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD DATABASE
# ============================================================

def load_system():

    print(
        "Loading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        "Loading Indian Standards index..."
    )

    standard_index = faiss.read_index(
        str(STANDARD_INDEX)
    )

    standard_metadata = load_json(
        STANDARD_METADATA
    )

    print(
        f"  Standards vectors : "
        f"{standard_index.ntotal}"
    )

    print(
        f"  Standards metadata: "
        f"{len(standard_metadata)}"
    )

    print(
        "Loading certification index..."
    )

    cert_index = faiss.read_index(
        str(CERT_INDEX)
    )

    cert_metadata = load_json(
        CERT_METADATA
    )

    print(
        f"  Certification vectors : "
        f"{cert_index.ntotal}"
    )

    print(
        f"  Certification metadata: "
        f"{len(cert_metadata)}"
    )

    return (
        model,
        standard_index,
        standard_metadata,
        cert_index,
        cert_metadata
    )


# ============================================================
# BUILD QUERY
# ============================================================

def build_query(requirement):

    parts = [
        f"Product Title: "
        f"{requirement['title']}",

        f"Description: "
        f"{requirement['description']}"
    ]

    if requirement["category"]:

        parts.append(
            f"Category: "
            f"{requirement['category']}"
        )

    if requirement["details"]:

        parts.append(
            f"Requirements: "
            f"{requirement['details']}"
        )

    return "\n".join(parts)


# ============================================================
# PRODUCT MATCHING
# ============================================================

def product_match_score(
    requirement,
    evidence_text
):

    product = normalize_text(
        requirement["title"]
    )

    description = normalize_text(
        requirement["description"]
    )

    evidence = normalize_text(
        evidence_text
    )

    score = 0.0

    # Exact product phrase
    if product and product in evidence:

        score += 20

    # Product token overlap
    product_tokens = tokenize(
        product
    )

    if product_tokens:

        matched = sum(
            1
            for token in product_tokens
            if token in evidence
        )

        score += (
            matched /
            len(product_tokens)
        ) * 10

    # Description overlap
    description_tokens = tokenize(
        description
    )

    if description_tokens:

        matched = sum(
            1
            for token in description_tokens
            if token in evidence
        )

        overlap = (
            matched /
            len(description_tokens)
        )

        score += overlap * 5

    return score


# ============================================================
# RETRIEVE STANDARDS
# ============================================================

def retrieve_standards(
    model,
    index,
    metadata,
    query
):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        embedding,
        TOP_K
    )

    candidates = []

    query_tokens = tokenize(
        query
    )

    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if idx < 0:
            continue

        item = metadata[idx]

        text = item.get(
            "text",
            ""
        )

        standard_numbers = item.get(
            "standard_numbers",
            []
        )

        if not standard_numbers:

            standard_numbers = (
                extract_standard_numbers(
                    text
                )
            )

        lexical_overlap = len(
            query_tokens.intersection(
                tokenize(text)
            )
        )

        product_score = product_match_score(
            CURRENT_REQUIREMENT,
            text
        )

        final_score = (
            float(score) * 10
            + lexical_overlap * 0.15
            + product_score
        )

        candidates.append({
            "rank": rank,
            "semantic_score": float(score),
            "lexical_overlap": lexical_overlap,
            "product_score": product_score,
            "final_score": final_score,
            "metadata": item,
            "standard_numbers": standard_numbers
        })

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return candidates


# ============================================================
# PAGE RESOLUTION
# ============================================================

def resolve_page(metadata):

    page = metadata.get(
        "page_number"
    )

    if page not in [
        None,
        "",
        0
    ]:

        return page

    source_pdf = metadata.get(
        "source_pdf"
    )

    if not source_pdf:
        return None

    pdf_path = (
        STANDARD_PDF_DIR /
        source_pdf
    )

    if not pdf_path.exists():

        pdf_path = (
            BASE_DIR /
            source_pdf
        )

    if not pdf_path.exists():
        return None

    chunk_text = metadata.get(
        "text",
        ""
    )

    if not chunk_text:
        return None

    try:

        document = pymupdf.open(
            str(pdf_path)
        )

        # ----------------------------------------------------
        # First try the exact standard number.
        # ----------------------------------------------------

        standards = metadata.get(
            "standard_numbers",
            []
        )

        standard_patterns = []

        for standard in standards:

            match = re.search(
                r"IS\s+(\d{4,6})"
                r"(?:\s*\([^)]{1,100}\))?"
                r"\s*:\s*(\d{4})",
                standard,
                flags=re.IGNORECASE
            )

            if match:

                standard_patterns.append(
                    re.compile(
                        r"\bIS\s+"
                        + re.escape(
                            match.group(1)
                        )
                        + r"\s*:\s*"
                        + re.escape(
                            match.group(2)
                        )
                        + r"\b",
                        flags=re.IGNORECASE
                    )
                )

        # ----------------------------------------------------
        # Normalize chunk for fallback matching.
        # ----------------------------------------------------

        normalized_chunk = normalize_text(
            chunk_text
        )

        # ----------------------------------------------------
        # Search every PDF page.
        # ----------------------------------------------------

        for page_number, page in enumerate(
            document,
            start=1
        ):

            page_raw = page.get_text()

            page_normalized = normalize_text(
                page_raw
            )

            # 1. Standard-number match
            for pattern in standard_patterns:

                if pattern.search(page_raw):

                    document.close()

                    return page_number

            # 2. Chunk beginning match
            chunk_start = normalized_chunk[:300]

            if (
                len(chunk_start) >= 80
                and chunk_start in page_normalized
            ):

                document.close()

                return page_number

        document.close()

    except Exception:

        pass

    return None


# ============================================================
# STANDARD RELATIONSHIP
# ============================================================

def classify_relationship(
    requirement,
    evidence
):

    combined_text = " ".join(
        item["metadata"].get(
            "text",
            ""
        )
        for item in evidence
    )

    text = normalize_text(
        combined_text
    )

    product_title = normalize_text(
        requirement["title"]
    )

    # --------------------------------------------------------
    # DIRECT PRODUCT MATCH
    # --------------------------------------------------------

    if (
        product_title
        and product_title in text
    ):

        return "Directly Applicable"

    title_tokens = tokenize(
        requirement["title"]
    )

    if title_tokens:

        matched = sum(
            1
            for token in title_tokens
            if token in text
        )

        ratio = (
            matched /
            len(title_tokens)
        )

        if ratio >= 0.75:

            return "Directly Applicable"

    # --------------------------------------------------------
    # REFERENCED STANDARD
    # --------------------------------------------------------

    reference_phrases = [
        "referred to",
        "reference standard",
        "referenced standard",
        "shall comply with",
        "in accordance with"
    ]

    if any(
        phrase in text
        for phrase in reference_phrases
    ):

        return "Referenced"

    # --------------------------------------------------------
    # TESTING STANDARD
    # --------------------------------------------------------

    testing_phrases = [
        "test method",
        "test procedure",
        "testing method",
        "method of test",
        "determination of"
    ]

    if any(
        phrase in text
        for phrase in testing_phrases
    ):

        return "Testing"

    # --------------------------------------------------------
    # SAFETY STANDARD
    # --------------------------------------------------------

    safety_phrases = [
        "safety requirement",
        "safety requirements",
        "hazard",
        "flammability",
        "toxic substance"
    ]

    if any(
        phrase in text
        for phrase in safety_phrases
    ):

        return "Safety"

    return None


# ============================================================
# GROUP STANDARD EVIDENCE
# ============================================================

def group_standards(
    requirement,
    candidates
):

    groups = defaultdict(list)

    for candidate in candidates:

        for standard_number in (
            candidate["standard_numbers"]
        ):

            groups[
                standard_number
            ].append(candidate)

    valid_groups = {}

    for standard_number, evidence in (
        groups.items()
    ):

        relationship = classify_relationship(
            requirement,
            evidence
        )

        if relationship is None:
            continue

        valid_groups[
            standard_number
        ] = {
            "relationship": relationship,
            "evidence": evidence
        }

    return valid_groups


# ============================================================
# CERTIFICATION RETRIEVAL
# ============================================================

def retrieve_certification(
    model,
    index,
    metadata,
    query
):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    embedding = np.asarray(
        embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        embedding,
        TOP_K
    )

    candidates = []

    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if idx < 0:
            continue

        candidates.append({
            "rank": rank,
            "score": float(score),
            "metadata": metadata[idx]
        })

    return candidates


# ============================================================
# CERTIFICATION EVIDENCE CHECK
# ============================================================

def check_certification_evidence(
    requirement,
    standard_number,
    candidates
):

    if not standard_number:
        return []

    product_title = normalize_text(
        requirement["title"]
    )

    matches = []

    for candidate in candidates:

        text = normalize_text(
            candidate["metadata"].get(
                "text",
                ""
            )
        )

        has_product = (
            product_title
            and product_title in text
        )

        # ----------------------------------------------------
        # Standard number matching
        # Handles:
        # IS 17873:2022
        # IS 17873 : 2022
        # ----------------------------------------------------

        standard_match = re.search(
            r"IS\s+(\d{4,6})"
            r"(?:\s*\([^)]{1,100}\))?"
            r"\s*:\s*(\d{4})",
            standard_number,
            flags=re.IGNORECASE
        )

        has_standard = False

        if standard_match:

            standard_pattern = re.compile(
                r"\bIS\s+"
                + re.escape(
                    standard_match.group(1)
                )
                + r"\s*:\s*"
                + re.escape(
                    standard_match.group(2)
                )
                + r"\b",
                flags=re.IGNORECASE
            )

            has_standard = bool(
                standard_pattern.search(text)
            )

        certification_terms = [
            "compulsory certification",
            "mandatory certification",
            "standard mark",
            "isi mark",
            "licence",
            "license",
            "certification"
        ]

        has_certification_term = any(
            term in text
            for term in certification_terms
        )

        if (
            has_product
            and has_standard
            and has_certification_term
        ):

            matches.append(
                candidate
            )

    return matches


# ============================================================
# EVIDENCE DISPLAY
# ============================================================

def display_evidence(
    evidence,
    maximum=1
):

    shown = set()
    count = 0

    evidence = sorted(
        evidence,
        key=lambda x: x.get(
            "final_score",
            0
        ),
        reverse=True
    )

    for item in evidence:

        metadata = item["metadata"]

        source = metadata.get(
            "source_pdf",
            "Unknown source"
        )

        page = resolve_page(
            metadata
        )

        text = re.sub(
            r"\s+",
            " ",
            metadata.get(
                "text",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # Remove duplicate source/page entries
        # ----------------------------------------------------

        key = (
            source,
            page
        )

        if key in shown:
            continue

        shown.add(key)

        print(
            f"\n  Source PDF : {source}"
        )

        print(
            f"  Page       : "
            f"{page if page else 'Not available'}"
        )

        if len(text) > 600:

            text = (
                text[:600]
                + "..."
            )

        print(
            f"  Evidence   : {text}"
        )

        count += 1

        if count >= maximum:
            break


# ============================================================
# SELECT PRIMARY STANDARD
# ============================================================

def select_primary_standard(
    standard_groups
):

    if not standard_groups:
        return None

    priority = {
        "Directly Applicable": 4,
        "Referenced": 3,
        "Testing": 2,
        "Safety": 1
    }

    ranked = []

    for number, data in (
        standard_groups.items()
    ):

        relationship = data[
            "relationship"
        ]

        best_evidence = max(
            data["evidence"],
            key=lambda x: x.get(
                "final_score",
                0
            )
        )

        score = (
            priority.get(
                relationship,
                0
            ) * 100
            + best_evidence.get(
                "final_score",
                0
            )
        )

        ranked.append(
            (
                score,
                number,
                data
            )
        )

    ranked.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    _, number, data = ranked[0]

    return {
        "standard_number": number,
        "relationship": data[
            "relationship"
        ],
        "evidence": data[
            "evidence"
        ]
    }


# ============================================================
# CONFIDENCE
# ============================================================

def calculate_confidence(primary):

    if not primary:
        return 0

    relationship = primary[
        "relationship"
    ]

    best_evidence = max(
        primary["evidence"],
        key=lambda x: x.get(
            "final_score",
            0
        )
    )

    metadata = best_evidence[
        "metadata"
    ]

    text = metadata.get(
        "text",
        ""
    )

    # --------------------------------------------------------
    # PRODUCT MATCH STRENGTH
    # --------------------------------------------------------

    product_score = product_match_score(
        CURRENT_REQUIREMENT,
        text
    )

    product_strength = min(
        product_score / 35.0,
        1.0
    )

    # --------------------------------------------------------
    # RELATIONSHIP STRENGTH
    # --------------------------------------------------------

    relationship_score = {
        "Directly Applicable": 1.00,
        "Referenced": 0.82,
        "Testing": 0.72,
        "Safety": 0.65
    }.get(
        relationship,
        0.50
    )

    # --------------------------------------------------------
    # EVIDENCE STRENGTH
    # --------------------------------------------------------

    evidence_strength = min(
        len(primary["evidence"]) / 3.0,
        1.0
    )

    # --------------------------------------------------------
    # FINAL CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        product_strength * 0.55
        + relationship_score * 0.30
        + evidence_strength * 0.15
    )

    return round(
        confidence * 100
    )


# ============================================================
# STAGE 5 OUTPUT
# ============================================================

def generate_report(
    requirement,
    standard_groups,
    certification_candidates
):

    primary = select_primary_standard(
        standard_groups
    )

    print("\n")
    print(
        "STAGE 5 OF 8  •  OUTPUT"
    )

    # ========================================================
    # PRIMARY STANDARD
    # ========================================================

    print(
        "\nRECOMMENDED INDIAN STANDARD"
    )

    if not primary:

        print(
            "\nNo directly applicable Indian Standard "
            "was established from the retrieved BIS evidence."
        )

    else:

        number = primary[
            "standard_number"
        ]

        relationship = primary[
            "relationship"
        ]

        title = extract_standard_title(
            primary["evidence"],
            number
        )

        print(
            f"\n{number}"
        )

        if title:

            print(
                title
            )

        print(
            "\nRelationship"
        )

        print(
            f"  {relationship}"
        )

        # ----------------------------------------------------
        # MATCH CONFIDENCE
        # ----------------------------------------------------

        confidence = calculate_confidence(
            primary
        )

        print(
            "\nMatch Confidence"
        )

        print(
            f"  {confidence}%"
        )

        # ----------------------------------------------------
        # WHY THIS STANDARD
        # ----------------------------------------------------

        print(
            "\nWHY THIS STANDARD"
        )

        if relationship == "Directly Applicable":

            print(
                f"  {number} was identified as directly "
                f"applicable to the procurement requirement "
                f"from the retrieved BIS evidence."
            )

        elif relationship == "Referenced":

            print(
                f"  {number} was identified as a referenced "
                f"standard in the retrieved BIS evidence."
            )

        elif relationship == "Testing":

            print(
                f"  {number} was identified as a testing-related "
                f"standard in the retrieved BIS evidence."
            )

        elif relationship == "Safety":

            print(
                f"  {number} was identified as a safety-related "
                f"standard in the retrieved BIS evidence."
            )

        # ----------------------------------------------------
        # SOURCE EVIDENCE
        # ----------------------------------------------------

        print(
            "\nSOURCE EVIDENCE"
        )

        display_evidence(
            primary["evidence"],
            maximum=1
        )

    # ========================================================
    # RELATED STANDARDS
    # ========================================================

    print(
        "\nALIGNED & REFERENCED STANDARDS"
    )

    related = []

    for number, data in (
        standard_groups.items()
    ):

        if (
            primary
            and number
            == primary["standard_number"]
        ):

            continue

        if data["relationship"] in [
            "Referenced",
            "Testing",
            "Safety"
        ]:

            related.append(
                (
                    number,
                    data
                )
            )

    if not related:

        print(
            "\n  No additional standards were established "
            "as meaningfully related to this procurement."
        )

    else:

        for number, data in related:

            title = extract_standard_title(
                data["evidence"],
                number
            )

            print(
                f"\n{number}"
            )

            if title:

                print(
                    title
                )

            print(
                f"  Relationship: "
                f"{data['relationship']}"
            )

            best = max(
                data["evidence"],
                key=lambda x: x.get(
                    "final_score",
                    0
                )
            )

            metadata = best[
                "metadata"
            ]

            page = resolve_page(
                metadata
            )

            print(
                f"  Source      : "
                f"{metadata.get('source_pdf', 'Unknown')}"
            )

            print(
                f"  Page        : "
                f"{page if page else 'Not available'}"
            )

    # ========================================================
    # TECHNICAL SPECIFICATIONS
    # ========================================================

    print(
        "\nTECHNICAL SPECIFICATIONS"
    )

    if requirement["details"]:

        requirements = split_requirements(
            requirement["details"]
        )

        for item in requirements:

            print(
                f"\n  {item}"
            )

    else:

        print(
            "\n  No additional technical requirements "
            "were explicitly provided."
        )

    # ========================================================
    # SAFETY MEASURES
    # ========================================================

    print(
        "\nSAFETY MEASURES"
    )

    safety_standards = []

    for number, data in (
        standard_groups.items()
    ):

        if data["relationship"] == "Safety":

            safety_standards.append(
                (
                    number,
                    data
                )
            )

    if safety_standards:

        for number, data in safety_standards:

            print(
                f"\n  ✓ Safety requirements identified "
                f"under {number}."
            )

            best = max(
                data["evidence"],
                key=lambda x: x.get(
                    "final_score",
                    0
                )
            )

            metadata = best[
                "metadata"
            ]

            page = resolve_page(
                metadata
            )

            print(
                f"    Source : "
                f"{metadata.get('source_pdf', 'Unknown')}"
            )

            print(
                f"    Page   : "
                f"{page if page else 'Not available'}"
            )

            evidence_text = re.sub(
                r"\s+",
                " ",
                metadata.get(
                    "text",
                    ""
                )
            ).strip()

            if len(evidence_text) > 350:

                evidence_text = (
                    evidence_text[:350]
                    + "..."
                )

            print(
                f"    Evidence: "
                f"{evidence_text}"
            )

    else:

        print(
            "\n  No product-specific safety requirements "
            "were established from the retrieved BIS evidence."
        )

    # ========================================================
    # IMPLEMENTATION MEASURES
    # ========================================================

    print(
        "\nIMPLEMENTATION MEASURES"
    )

    if primary:

        print(
            f"\n  ✓ Reference "
            f"{primary['standard_number']} "
            f"in the procurement specification."
        )

        print(
            "\n  ✓ Ensure the supplied product is evaluated "
            "against the applicable requirements of the "
            "identified Indian Standard."
        )

        print(
            "\n  ✓ Check relevant test reports, inspection "
            "records or conformity documents where required."
        )

        print(
            "\n  ✓ Verify that the supplied product matches "
            "the material, workmanship and physical "
            "requirements specified in the procurement."
        )

    else:

        print(
            "\n  No standard-specific implementation measures "
            "could be established because no directly "
            "applicable Indian Standard was identified."
        )

    # ========================================================
    # CERTIFICATION REQUIREMENTS
    # ========================================================

    print(
        "\nCERTIFICATION REQUIREMENTS"
    )

    certification_matches = []

    if primary:

        certification_matches = (
            check_certification_evidence(
                requirement,
                primary["standard_number"],
                certification_candidates
            )
        )

    if certification_matches:

        print(
            "\n  Product-specific BIS certification "
            "evidence was retrieved."
        )

        for match in certification_matches[:3]:

            metadata = match[
                "metadata"
            ]

            page = metadata.get(
                "page_number"
            )

            print(
                f"\n  Source PDF : "
                f"{metadata.get('source_pdf', 'Unknown')}"
            )

            print(
                f"  Page       : "
                f"{page if page else 'Not available'}"
            )

            text = re.sub(
                r"\s+",
                " ",
                metadata.get(
                    "text",
                    ""
                )
            ).strip()

            if len(text) > 500:

                text = (
                    text[:500]
                    + "..."
                )

            print(
                f"  Evidence   : "
                f"{text}"
            )

    else:

        print(
            "\n  BIS Product Certification"
        )

        print(
            "    Not established from available "
            "product-specific BIS evidence."
        )

        print(
            "\n  CRS"
        )

        print(
            "    Not established from available "
            "product-specific BIS evidence."
        )

        print(
            "\n  Hallmarking"
        )

        print(
            "    Not established from available "
            "product-specific BIS evidence."
        )

    # ========================================================
    # FINAL RECOMMENDATION
    # ========================================================

    print(
        "\nFINAL RECOMMENDATION"
    )

    if primary:

        number = primary[
            "standard_number"
        ]

        relationship = primary[
            "relationship"
        ]

        if relationship == "Directly Applicable":

            print(
                f"\n  Use {number} as the primary "
                f"Indian Standard reference for "
                f"this procurement."
            )

        else:

            print(
                f"\n  {number} was identified as a "
                f"{relationship.lower()} standard, "
                f"but the available evidence does not "
                f"establish it as directly applicable."
            )

    else:

        print(
            "\n  No Indian Standard could be established "
            "as directly applicable from the available evidence."
        )

    if certification_matches:

        print(
            "\n  Certification status:"
        )

        print(
            "    Product-specific evidence found."
        )

    else:

        print(
            "\n  Certification status:"
        )

        print(
            "    Not established from available BIS evidence."
        )

    confidence = calculate_confidence(
        primary
    )

    print(
        "\nRecommendation Confidence"
    )

    print(
        f"  {confidence}%"
    )

    print(
        "\nEnd of recommendation."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # GET REQUIREMENT
    # --------------------------------------------------------

    CURRENT_REQUIREMENT = (
        get_user_requirement()
    )

    # --------------------------------------------------------
    # STAGE 2 — REQUIREMENT VERIFICATION
    # --------------------------------------------------------

    display_requirement_verification(
        CURRENT_REQUIREMENT
    )

    # --------------------------------------------------------
    # LOAD SYSTEM
    # --------------------------------------------------------

    (
        model,
        standard_index,
        standard_metadata,
        cert_index,
        cert_metadata
    ) = load_system()

    # --------------------------------------------------------
    # BUILD QUERY
    # --------------------------------------------------------

    query = build_query(
        CURRENT_REQUIREMENT
    )

    # --------------------------------------------------------
    # STANDARD RETRIEVAL
    # --------------------------------------------------------

    print(
        "\nSearching Indian Standards..."
    )

    standard_candidates = retrieve_standards(
        model,
        standard_index,
        standard_metadata,
        query
    )

    # --------------------------------------------------------
    # GROUP + FILTER
    # --------------------------------------------------------

    standard_groups = group_standards(
        CURRENT_REQUIREMENT,
        standard_candidates
    )

    # --------------------------------------------------------
    # CERTIFICATION RETRIEVAL
    # --------------------------------------------------------

    print(
        "Checking certification evidence..."
    )

    certification_candidates = (
        retrieve_certification(
            model,
            cert_index,
            cert_metadata,
            query
        )
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    generate_report(
        CURRENT_REQUIREMENT,
        standard_groups,
        certification_candidates
    )