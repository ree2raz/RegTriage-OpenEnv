"""
redact.py — PII redaction for transcript text.

Simulates the output of production PII redaction pipelines (AWS Comprehend,
Google DLP, Azure Presidio). In real contact center analytics, transcripts
are ALWAYS scrubbed of Personally Identifiable Information before they reach
any ML/RL training system.

This changes what the LLM agent sees as observations:
  Before: "Hi, my name is Maria Lopez. My account is 5678-1234."
  After:  "Hi, my name is [CUSTOMER_NAME]. My account is [ACCOUNT_NUMBER]."

Key design decisions:
  - Dollar amounts are PRESERVED (compliance-relevant for unauthorized commitments)
  - Percentages are PRESERVED (compliance-relevant for rate quoting)
  - Named entity redaction requires an explicit dict (no NER in MVP)
  - Regex-based for structured PII (accounts, SSN, phone, email)
  - Already-redacted tokens (e.g., [ACCOUNT_NUMBER]) are left untouched
"""

import re
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# Regex Patterns — ordered from most specific to least specific
# to avoid partial matches
# ══════════════════════════════════════════════════════════════════

# Full SSN: 123-45-6789 or 123 45 6789
_SSN_FULL = re.compile(r'\b\d{3}[-\s]\d{2}[-\s]\d{4}\b')

# Last-4 SSN: contextual pattern — digits after SSN/social/security keywords
_SSN_LAST4 = re.compile(
    r'(?:SSN|social(?:\s+security)?(?:\s+number)?|last\s+four(?:\s+digits)?(?:\s+of\s+(?:my|your|the)\s+)?'
    r'(?:SSN|social(?:\s+security)?(?:\s+number)?)?)[\s:]*(?:is\s+|are\s+)?(\d{4})\b',
    re.IGNORECASE,
)

# Account numbers: 2+ groups of 4 digits separated by dashes or spaces
# (e.g., 1234-5678, 1234 5678 9012, 4532-1234-5678-9012)
_ACCOUNT_NUMBER = re.compile(r'\b\d{4}(?:[-\s]\d{4}){1,3}\b')

# Phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567
_PHONE = re.compile(
    r'(?:\(\d{3}\)\s*|\b\d{3}[-.])\d{3}[-.]?\d{4}\b'
)

# Email addresses
_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')


def redact_pii(text: str, entity_names: Optional[dict[str, str]] = None) -> str:
    """
    Redact PII from transcript text.

    Applies regex-based redaction for structured PII (account numbers, SSN,
    phone numbers, email addresses), then applies named entity replacement
    if an entity_names dict is provided.

    Args:
        text:          Raw transcript utterance text
        entity_names:  Optional mapping of names → replacement tokens,
                       e.g., {"Maria Lopez": "[CUSTOMER_NAME]",
                              "Maria": "[CUSTOMER_NAME]",
                              "Sarah": "[AGENT_NAME]"}
                       Longer names are matched first to avoid partial replacement.

    Returns:
        Text with PII replaced by tagged tokens. Dollar amounts and
        percentages are preserved (compliance-relevant).
    """
    if not text:
        return text

    # Skip already-redacted tokens
    if not any(c.isdigit() or '@' in text for c in text):
        # Quick exit: if no digits and no @, only names could match
        if entity_names:
            return _redact_names(text, entity_names)
        return text

    # ── Order matters: most specific patterns first ──

    # 1. Full SSN (before account numbers, since SSN has dashes too)
    text = _SSN_FULL.sub('[SSN_REDACTED]', text)

    # 2. Contextual last-4 SSN (before generic digit removal)
    text = _SSN_LAST4.sub(_replace_ssn_last4, text)

    # 3. Phone numbers (before account numbers, to avoid 555-123-4567 matching as account)
    text = _PHONE.sub('[PHONE_REDACTED]', text)

    # 4. Account numbers (digit groups with dashes)
    text = _ACCOUNT_NUMBER.sub('[ACCOUNT_NUMBER]', text)

    # 5. Email addresses
    text = _EMAIL.sub('[EMAIL_REDACTED]', text)

    # 6. Named entities (last, so they don't interfere with structural patterns)
    if entity_names:
        text = _redact_names(text, entity_names)

    return text


def _replace_ssn_last4(match: re.Match) -> str:
    """Replace only the 4-digit portion, keeping the context prefix."""
    full = match.group(0)
    digits = match.group(1)
    return full.replace(digits, '[SSN_REDACTED]')


def _redact_names(text: str, entity_names: dict[str, str]) -> str:
    """Replace named entities in text, longest match first."""
    # Sort by length descending — "Maria Lopez" before "Maria"
    sorted_names = sorted(entity_names.keys(), key=len, reverse=True)
    for name in sorted_names:
        replacement = entity_names[name]
        # Case-insensitive replacement
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub(replacement, text)
    return text
