"""
tests/test_redact.py — Tests for PII redaction module.

Tests written BEFORE implementation (TDD). The redact_pii function
simulates what production PII redaction pipelines (AWS Comprehend,
Google DLP, Azure Presidio) produce: tagged replacement tokens in
place of sensitive data.

Contract:
    redact_pii(text, entity_names=None) -> str

    - text:          Raw transcript utterance
    - entity_names:  Optional dict mapping names to replacement tokens,
                     e.g., {"Maria Lopez": "[CUSTOMER_NAME]", "Sarah": "[AGENT_NAME]"}
    - Returns:       Text with PII replaced by tagged tokens
"""

import pytest
from redact import redact_pii


# ══════════════════════════════════════════════════════════════════
# 1. Account Numbers — dash-separated digit groups
# ══════════════════════════════════════════════════════════════════

class TestAccountNumbers:
    """Account numbers appear as dash-separated digit groups (e.g., 1234-5678-9012)."""

    def test_standard_account_number(self):
        text = "My account number is 1234-5678-9012."
        assert redact_pii(text) == "My account number is [ACCOUNT_NUMBER]."

    def test_account_number_four_groups(self):
        text = "It's 4532-1234-5678-9012, I think."
        assert redact_pii(text) == "It's [ACCOUNT_NUMBER], I think."

    def test_account_number_two_groups(self):
        text = "Account 5678-1234 please."
        assert redact_pii(text) == "Account [ACCOUNT_NUMBER] please."

    def test_account_number_no_context(self):
        """Standalone digit groups should still be redacted."""
        text = "Sure, 9876-5432-1098."
        assert redact_pii(text) == "Sure, [ACCOUNT_NUMBER]."

    def test_year_not_redacted(self):
        """4-digit standalone numbers (like years) should NOT be redacted."""
        text = "This was opened in 2024."
        assert redact_pii(text) == "This was opened in 2024."

    def test_space_separated_account_number(self):
        """ASR often produces spaces instead of dashes in account numbers."""
        text = "It's 1234 5678 9012."
        assert redact_pii(text) == "It's [ACCOUNT_NUMBER]."


# ══════════════════════════════════════════════════════════════════
# 2. Social Security Numbers
# ══════════════════════════════════════════════════════════════════

class TestSSN:
    """SSNs appear as full (123-45-6789) or last-4 references."""

    def test_full_ssn(self):
        text = "My social is 123-45-6789."
        assert redact_pii(text) == "My social is [SSN_REDACTED]."

    def test_last_four_ssn(self):
        """Last 4 digits of SSN, commonly given as a 4-digit number after context words."""
        text = "The last four of my SSN is 4532."
        result = redact_pii(text)
        assert "[SSN_REDACTED]" in result
        assert "4532" not in result

    def test_last_four_social(self):
        text = "last four digits of your social security number is 1234"
        result = redact_pii(text)
        assert "[SSN_REDACTED]" in result
        assert "1234" not in result

    def test_ssn_with_spaces(self):
        text = "SSN 123 45 6789 on file."
        assert "[SSN_REDACTED]" in redact_pii(text)


# ══════════════════════════════════════════════════════════════════
# 3. Phone Numbers
# ══════════════════════════════════════════════════════════════════

class TestPhoneNumbers:
    """Phone numbers in various US formats."""

    def test_parenthetical_format(self):
        text = "Call me at (555) 123-4567."
        assert redact_pii(text) == "Call me at [PHONE_REDACTED]."

    def test_dash_format(self):
        text = "My number is 555-123-4567."
        assert redact_pii(text) == "My number is [PHONE_REDACTED]."

    def test_dot_format(self):
        text = "Reach me at 555.123.4567."
        assert redact_pii(text) == "Reach me at [PHONE_REDACTED]."


# ══════════════════════════════════════════════════════════════════
# 4. Email Addresses
# ══════════════════════════════════════════════════════════════════

class TestEmail:
    def test_standard_email(self):
        text = "My email is john.doe@example.com."
        assert redact_pii(text) == "My email is [EMAIL_REDACTED]."

    def test_email_mid_sentence(self):
        text = "Send it to maria_lopez99@gmail.com please."
        assert redact_pii(text) == "Send it to [EMAIL_REDACTED] please."


# ══════════════════════════════════════════════════════════════════
# 5. Named Entities (customer/agent names)
# ══════════════════════════════════════════════════════════════════

class TestNamedEntities:
    """Names are redacted via an explicit entity_names dict, not NER."""

    def test_customer_full_name(self):
        entities = {"Maria Lopez": "[CUSTOMER_NAME]"}
        text = "Hi, this is Maria Lopez."
        assert redact_pii(text, entity_names=entities) == "Hi, this is [CUSTOMER_NAME]."

    def test_customer_first_name_only(self):
        entities = {"Maria Lopez": "[CUSTOMER_NAME]", "Maria": "[CUSTOMER_NAME]"}
        text = "Thanks, Maria, let me check that."
        assert redact_pii(text, entity_names=entities) == "Thanks, [CUSTOMER_NAME], let me check that."

    def test_agent_name(self):
        entities = {"Sarah": "[AGENT_NAME]"}
        text = "My name is Sarah, I'll be assisting you."
        assert redact_pii(text, entity_names=entities) == "My name is [AGENT_NAME], I'll be assisting you."

    def test_no_entities_no_name_redaction(self):
        """Without entity_names, names are NOT redacted (no NER in MVP)."""
        text = "Hi, this is Maria Lopez."
        assert redact_pii(text) == "Hi, this is Maria Lopez."

    def test_case_insensitive_name_match(self):
        entities = {"john chen": "[CUSTOMER_NAME]"}
        text = "Hello John Chen, welcome back."
        result = redact_pii(text, entity_names=entities)
        assert "[CUSTOMER_NAME]" in result

    def test_multiple_entities(self):
        entities = {"Maria": "[CUSTOMER_NAME]", "Brian": "[AGENT_NAME]"}
        text = "Brian told Maria the balance was updated."
        result = redact_pii(text, entity_names=entities)
        assert "[AGENT_NAME]" in result
        assert "[CUSTOMER_NAME]" in result
        assert "Brian" not in result
        assert "Maria" not in result


# ══════════════════════════════════════════════════════════════════
# 6. Dollar Amounts — MUST NOT be redacted
# ══════════════════════════════════════════════════════════════════

class TestDollarAmountsPreserved:
    """Dollar amounts are compliance-relevant and must be kept."""

    def test_dollar_amount_preserved(self):
        text = "The fee is $45.99."
        assert redact_pii(text) == "The fee is $45.99."

    def test_large_amount_preserved(self):
        text = "Your balance is $12,345.67."
        assert redact_pii(text) == "Your balance is $12,345.67."

    def test_rate_preserved(self):
        text = "I can offer you 3.95% interest."
        assert redact_pii(text) == "I can offer you 3.95% interest."


# ══════════════════════════════════════════════════════════════════
# 7. Edge Cases
# ══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_string(self):
        assert redact_pii("") == ""

    def test_no_pii(self):
        text = "Thank you for calling. How can I help?"
        assert redact_pii(text) == "Thank you for calling. How can I help?"

    def test_multiple_pii_types(self):
        entities = {"John": "[CUSTOMER_NAME]"}
        text = "John, your account 1234-5678 and SSN ending 4532."
        result = redact_pii(text, entity_names=entities)
        assert "[CUSTOMER_NAME]" in result
        assert "[ACCOUNT_NUMBER]" in result
        assert "John" not in result
        assert "1234-5678" not in result

    def test_redaction_idempotent(self):
        """Redacting an already-redacted string should not double-redact."""
        text = "Account [ACCOUNT_NUMBER] confirmed."
        assert redact_pii(text) == "Account [ACCOUNT_NUMBER] confirmed."
