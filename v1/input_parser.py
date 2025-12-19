# input_parser.py
import re
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Core text utilities
# ----------------------------

SECTION_HEADINGS = {
    "skills": [
        r"skills",
        r"technical skills",
        r"core competencies",
        r"tools",
        r"technologies",
    ],
    "experience": [
        r"professional experience",
        r"work experience",
        r"experience",
        r"employment",
        r"industry experience",
    ],
    "education": [
        r"education",
        r"academic background",
        r"qualifications",
        r"certifications",
        r"certification",
        r"training",
    ],
}

JD_QUAL_HEADINGS = [
    r"desired qualifications",
    r"qualifications",
    r"requirements",
    r"what we(?:'|’)re looking for",
    r"what you bring",
]

BULLET_CHARS = r"[\u2022\u25CF\u25AA\u25AB\u25E6\-\*\u2013\u2014]"


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines down to 2 (helps section parsing)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in _clean(text).splitlines()]


def _first_nonempty_lines(text: str, n: int = 10) -> List[str]:
    return [ln for ln in _lines(text) if ln][:n]


def _is_heading_line(line: str) -> bool:
    """
    Heuristic to detect section headings in resumes/JDs.
    Works for:
      - "Skills"
      - "Professional Experience"
      - "Desired Qualifications"
      - "Skills:" etc.
    """
    if not line:
        return False
    # Very short lines are often headings
    if len(line) <= 2:
        return False

    # Common heading patterns: Title Case / ALL CAPS / ends with ':'
    if line.endswith(":"):
        return True

    # All-caps headings (allow spaces and punctuation)
    if re.fullmatch(r"[A-Z0-9][A-Z0-9 &/\-\.,]{2,}", line):
        return True

    # Title-like lines with 1–5 words (avoid normal sentences)
    words = line.split()
    if 1 <= len(words) <= 6 and all(re.fullmatch(r"[A-Za-z][A-Za-z&/\-\.]*", w) for w in words):
        # Avoid lines that look like regular sentences
        if not re.search(r"[.!?]$", line):
            return True

    return False


def _heading_regex(heading_variants: List[str]) -> re.Pattern:
    """
    Build a heading matcher that triggers at line start, case-insensitive,
    allowing optional trailing ':'.
    """
    joined = "|".join(f"(?:{h})" for h in heading_variants)
    return re.compile(rf"(?im)^(?:{joined})\s*:?\s*$")


def _find_section_span(text: str, heading_variants: List[str]) -> Optional[Tuple[int, int]]:
    """
    Find a section's (start, end) character span using line-based headings.

    Strategy:
      1) Find heading line.
      2) Start content after that heading line.
      3) End at the next heading line (or end of text).
    """
    t = _clean(text)
    if not t:
        return None

    # Find heading line location
    hre = _heading_regex(heading_variants)
    m = hre.search(t)
    if not m:
        return None

    start = m.end()

    # Find next heading after start
    rest = t[start:]
    # Identify headings by scanning lines and using _is_heading_line
    # but only treat it as a heading if it is separated by a blank line or looks like a canonical heading
    offset = start
    lines = rest.splitlines(True)  # keep line endings
    idx = 0
    pos = offset
    started = False

    # Move to first non-empty after heading line
    while idx < len(lines) and not started:
        if lines[idx].strip() == "":
            pos += len(lines[idx])
            idx += 1
            continue
        started = True
        break

    content_start = pos

    # Now search for next heading line after content_start
    # We'll require: heading-like line AND either preceded by blank line OR ends with ':'
    # This avoids accidentally cutting on company names.
    pos2 = content_start
    prev_blank = True  # after heading, often blank line(s)
    for ln_with_end in lines[idx:]:
        ln = ln_with_end.strip()
        if ln == "":
            prev_blank = True
            pos2 += len(ln_with_end)
            continue

        if _is_heading_line(ln) and (prev_blank or ln.endswith(":")):
            # Stop BEFORE this heading line
            end = pos2
            return (content_start, end)

        prev_blank = False
        pos2 += len(ln_with_end)

    return (content_start, len(t))


def _normalize_bullets(block: str) -> str:
    """
    Normalize bullets and spacing while preserving readability.
    """
    b = block.strip()
    if not b:
        return ""

    # Normalize bullet glyphs to "• "
    b = re.sub(rf"(?m)^\s*{BULLET_CHARS}\s*", "• ", b)
    # Remove repeated spaces
    b = re.sub(r"[ \t]+", " ", b)
    # Clean excessive blank lines
    b = re.sub(r"\n{3,}", "\n\n", b)
    return b.strip()


def head_tail(text: str, head: int, tail: int) -> str:
    text = _clean(text)
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n...\n" + text[-tail:]


# ----------------------------
# Resume extraction
# ----------------------------

def extract_applicant_name(resume_text: str) -> str:
    """
    Improved:
      - Handles parentheses, middle initials, ALL CAPS, and names with 2–5 tokens
      - Avoids picking up address/email lines
    """
    lines = _first_nonempty_lines(resume_text, 12)

    # Filter out obvious non-name lines
    def is_contactish(ln: str) -> bool:
        return bool(
            re.search(r"@", ln)
            or re.search(r"\b\d{3}[\-\)\s]\d{3}[\-\s]\d{4}\b", ln)  # phone-ish
            or re.search(r"\b(st|street|ave|avenue|road|rd|ny|ca|tx|mi)\b", ln.lower())
            or "linkedin" in ln.lower()
            or "github" in ln.lower()
            or "|" in ln
        )

    candidates = [ln for ln in lines if not is_contactish(ln)]

    # Strong pattern: first line often is name
    if candidates:
        first = candidates[0].strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z'\-\.]*?(?:\s+\([A-Za-z][A-Za-z'\-\. ]+\))?(?:\s+[A-Za-z][A-Za-z'\-\.]*){1,4}", first):
            return first

    # Otherwise scan for best "name-like" line
    best = None
    best_score = -1
    for ln in candidates:
        ln2 = ln.strip()
        if len(ln2) > 60:
            continue
        # Must have at least two tokens and no digits
        if re.search(r"\d", ln2):
            continue
        tokens = ln2.replace(",", " ").split()
        if len(tokens) < 2 or len(tokens) > 6:
            continue
        # Score by how name-like the tokens are
        score = 0
        for t in tokens:
            if re.fullmatch(r"[A-Za-z][A-Za-z'\-\.]*", t):
                score += 2
            elif re.fullmatch(r"\([A-Za-z][A-Za-z'\-\. ]+\)", t):
                score += 1
            else:
                score -= 2
        if score > best_score:
            best_score = score
            best = ln2

    return best if best else (lines[0] if lines else "Applicant")


def extract_skills(resume_text: str) -> str:
    """
    Improved:
      - Prefer explicit Skills section spanning until next heading
      - Works with category-style skills ("X: a, b, c")
      - Falls back to compact keyword extraction if no section found
    """
    text = _clean(resume_text)

    span = _find_section_span(text, SECTION_HEADINGS["skills"])
    if span:
        block = text[span[0]:span[1]]
        block = _normalize_bullets(block)

        # If the block is huge, trim but keep complete lines
        lines = [ln for ln in block.splitlines() if ln.strip()]
        out_lines: List[str] = []
        total = 0
        for ln in lines:
            # avoid accidentally including next sections
            if _is_heading_line(ln) and total > 0:
                break
            out_lines.append(ln)
            total += len(ln) + 1
            if total >= 700:
                break
        return "\n".join(out_lines).strip()[:800]

    # Fallback: pull "skill-ish" tokens and common tech patterns
    # Keep this conservative to reduce junk words.
    candidates = re.findall(r"\b[A-Za-z][A-Za-z0-9\+\#\.\-/]{1,30}\b", text)
    stop = {
        "and","or","with","the","a","an","to","of","in","for","on","at","as","is","are",
        "from","by","this","that","it","be","was","were","will","can","may","using"
    }
    filtered: List[str] = []
    for c in candidates:
        cl = c.lower()
        if cl in stop:
            continue
        # Avoid very common resume words
        if cl in {"education","experience","skills","projects","research","intern","university"}:
            continue
        # Avoid long numeric-ish tokens
        if re.fullmatch(r"[A-Za-z]*\d+[A-Za-z0-9]*", c) and len(c) > 18:
            continue
        filtered.append(c)

    seen = set()
    uniq: List[str] = []
    for c in filtered:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
        if len(uniq) >= 45:
            break

    return ", ".join(uniq)


def extract_experience(resume_text: str) -> str:
    """
    Improved:
      - Extracts the Experience/Professional Experience section span
      - Returns the *most recent* role block when possible (top of section)
        because your field is "Current Working Experience"
    """
    text = _clean(resume_text)

    span = _find_section_span(text, SECTION_HEADINGS["experience"])
    if not span:
        # fallback: try to catch "Professional Experience" as a substring heading
        m = re.search(r"(?im)^\s*(professional experience|work experience|experience)\s*:?\s*$", text)
        if not m:
            return text[:1200]
        span = (m.end(), len(text))

    block = _normalize_bullets(text[span[0]:span[1]])
    if not block:
        return "See resume."

    # Heuristic: split by role headers like:
    # "Company Location" newline "Title Dates"
    # or "Company ... \n Role ... \n • bullets"
    # We'll attempt to extract the first "chunk" that looks like one role.
    lines = [ln for ln in block.splitlines() if ln.strip()]

    # If we can find a divider between roles (blank lines), use it.
    # Reconstruct with original blank lines preserved by re-splitting from raw block.
    raw_lines = block.splitlines()
    # Find first two blank-line breaks; role often is separated by a blank line
    chunks: List[str] = []
    cur: List[str] = []
    for ln in raw_lines:
        if ln.strip() == "":
            if cur:
                chunks.append("\n".join(cur).strip())
                cur = []
            continue
        cur.append(ln)
    if cur:
        chunks.append("\n".join(cur).strip())

    # Pick the first chunk that looks like a role (has bullets or dates)
    def looks_like_role(ch: str) -> bool:
        has_bullets = "• " in ch
        has_dates = bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", ch, re.I)) or bool(
            re.search(r"\b(20\d{2}|19\d{2})\b", ch)
        )
        # role chunks usually have at least 2 lines
        return (len([l for l in ch.splitlines() if l.strip()]) >= 2) and (has_bullets or has_dates)

    for ch in chunks:
        if looks_like_role(ch):
            return ch.strip()[:1400]

    # If no chunk matches, return trimmed whole section
    return block.strip()[:1400]


def extract_qualifications(resume_text: str) -> str:
    """
    Improved:
      - Prefer Education section
      - Otherwise use Certifications/Qualifications
      - Returns a compact readable block
    """
    text = _clean(resume_text)

    span = _find_section_span(text, SECTION_HEADINGS["education"])
    if span:
        block = _normalize_bullets(text[span[0]:span[1]])
        if block:
            # Keep it relatively short/clean
            lines = [ln for ln in block.splitlines() if ln.strip()]
            # Stop if we hit another obvious heading
            out: List[str] = []
            for ln in lines:
                if _is_heading_line(ln) and out:
                    break
                out.append(ln)
                if sum(len(x) for x in out) > 900:
                    break
            return "\n".join(out).strip()[:900]

    # fallback: sometimes education isn't labeled cleanly—try top portion
    top = "\n".join(_first_nonempty_lines(text, 25))
    # If top contains degree-like tokens, return it
    if re.search(r"\b(bachelor|master|phd|doctorate|b\.s\.|m\.s\.|bsc|msc)\b", top, re.I):
        return top[:900]

    return "See resume."


# ----------------------------
# JD extraction
# ----------------------------

def _extract_company_from_email_domain(jd_text: str) -> Optional[str]:
    """
    If JD contains an email like interns@numenta.com,
    infer company as the domain token (numenta).
    """
    t = _clean(jd_text)
    m = re.search(r"\b[A-Za-z0-9._%+\-]+@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b", t)
    if not m:
        return None
    domain = m.group(1).lower()
    # Take the registrable-ish leftmost label (not perfect but useful without hardcoding)
    parts = domain.split(".")
    if len(parts) >= 2:
        core = parts[-2]
        # Title-case it
        return core[:1].upper() + core[1:]
    return None


def extract_hiring_company(jd_text: str) -> str:
    """
    Improved:
      - Prefer 'X is looking for' / 'X is hiring' patterns
      - Else infer from email domain
      - Else fallback to first prominent proper noun-ish token frequency
    """
    text = _clean(jd_text)

    # Pattern: "Numenta is looking for ..."
    m = re.search(r"(?im)^\s*([A-Z][A-Za-z0-9&\-\.\s]{1,60}?)\s+is\s+(?:looking|hiring|seeking)\b", text)
    if m:
        cand = m.group(1).strip()
        # Avoid generic "We" / "About"
        if cand.lower() not in {"we", "about", "job", "position"} and len(cand) >= 2:
            return cand[:80]

    # Pattern: "at Company"
    m2 = re.search(r"(?i)\b(?:at|join)\s+([A-Z][A-Za-z0-9&\-\.\s]{2,60})(?:[,\.\n]|$)", text)
    if m2:
        cand = m2.group(1).strip()
        if len(cand.split()) <= 6:
            return cand[:80]

    # Email domain fallback
    dom = _extract_company_from_email_domain(text)
    if dom:
        return dom[:80]

    # Frequency-based fallback: find repeated Capitalized tokens (not sentence-start "The")
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9&\-\.\']{1,30}\b", text)
    ignore = {"The", "This", "How", "What", "About", "IMPORTANT", "NOTE"}
    freq = {}
    for tok in tokens:
        if tok in ignore:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    if freq:
        best = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))[0][0]
        return best[:80]

    return "Hiring Company"


def extract_job_title(jd_text: str) -> str:
    """
    Improved:
      - Prefer explicit title labels
      - Else infer from "looking for <ROLE>" patterns
      - Else use a strong title-ish first line fallback
    """
    text = _clean(jd_text)

    # Explicit label patterns
    m = re.search(r"(?im)^\s*(?:job title|title|role)\s*[:\-]\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip()[:100]

    # "is looking for <role>" pattern (captures role phrase)
    m2 = re.search(
        r"(?i)\bis\s+(?:looking|hiring|seeking)\s+for\s+(.+?)(?:\.\s|\n|\!|\:)",
        text,
    )
    if m2:
        role_phrase = m2.group(1).strip()
        # Clean common trailing qualifiers
        role_phrase = re.sub(r"(?i)\b(for the|for a|for an)\b.*$", "", role_phrase).strip()
        # If the phrase is long, keep the first chunk up to ~10 words
        words = role_phrase.split()
        if len(words) > 12:
            role_phrase = " ".join(words[:12])
        # Title-case-ish output is okay but don't force: keep original phrase
        return role_phrase[:100]

    # Common internship phrasing: "summer of 2026 internship opportunities"
    m3 = re.search(r"(?i)\bfor\s+the\s+(.+?)\s+internship", text)
    if m3:
        return m3.group(1).strip()[:100]

    # First line fallback (must look like a header, not "About the job")
    lines = _first_nonempty_lines(jd_text, 8)
    for ln in lines:
        ln2 = ln.strip()
        if ln2.lower() in {"about the job", "about", "job"}:
            continue
        if 3 <= len(ln2) <= 110 and not ln2.endswith("."):
            # Avoid lines that are obviously warnings/notes
            if "important note" in ln2.lower():
                continue
            return ln2[:100]

    return "Job Position"


# ----------------------------
# Orchestrator
# ----------------------------

def parse_resume_and_jd(resume_text: str, jd_text: str) -> Dict[str, str]:
    # Trim raw inputs BEFORE extraction to keep things fast
    resume_trim = head_tail(resume_text, head=3500, tail=1800)
    jd_trim = head_tail(jd_text, head=2500, tail=1200)

    applicant_name = extract_applicant_name(resume_trim)
    skills = extract_skills(resume_trim)
    experience = extract_experience(resume_trim)
    qualifications = extract_qualifications(resume_trim)

    job_title = extract_job_title(jd_trim)
    hiring_company = extract_hiring_company(jd_trim)

    return {
        "Job Title": job_title,
        "Hiring Company": hiring_company,
        "Applicant Name": applicant_name,
        "Skillsets": skills,
        "Current Working Experience": experience,
        "Qualifications": qualifications,
        # Cover Letter is what we generate, not input
    }
