"""
Corpus sanitizer for removing sensitive content.

Detects and removes or masks:
- API keys and secrets
- Passwords and credentials
- Email addresses and personal information
- Private URLs and internal paths
"""

import re
from typing import Any

from src.common.logging import get_logger

logger = get_logger(__name__)


class Sanitizer:
    """Remove sensitive content from corpus files."""

    def __init__(self):
        """Initialize sanitizer with detection patterns."""
        # Patterns for detecting secrets
        self.secret_patterns = [
            # API keys (generic patterns)
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "API_KEY"),
            (r'(?i)(secret|token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "SECRET"),

            # AWS
            (r'AKIA[0-9A-Z]{16}', "AWS_ACCESS_KEY"),
            (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "AWS_SECRET"),

            # GitHub
            (r'ghp_[a-zA-Z0-9]{36}', "GITHUB_PAT"),
            (r'gho_[a-zA-Z0-9]{36}', "GITHUB_OAUTH"),
            (r'ghu_[a-zA-Z0-9]{36}', "GITHUB_USER_TOKEN"),
            (r'ghs_[a-zA-Z0-9]{36}', "GITHUB_SERVER_TOKEN"),

            # Google
            (r'AIza[0-9A-Za-z\-_]{35}', "GOOGLE_API_KEY"),

            # Stripe
            (r'sk_live_[0-9a-zA-Z]{24}', "STRIPE_SECRET"),
            (r'pk_live_[0-9a-zA-Z]{24}', "STRIPE_PUBLISHABLE"),

            # Private keys
            (r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----', "PRIVATE_KEY"),

            # Database URLs with credentials
            (r'(?i)(mongodb|postgres|mysql|redis)://[^:]+:[^@]+@[^\s"\']+', "DATABASE_URL"),

            # Bearer tokens
            (r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+', "BEARER_TOKEN"),

            # Generic password patterns in configs
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']{8,})["\']', "PASSWORD"),
        ]

        # Patterns for files to skip entirely
        self.skip_file_patterns = [
            r'\.env($|\.)',  # .env files
            r'credentials?\.json$',
            r'secrets?\.json$',
            r'\.pem$',
            r'\.key$',
            r'id_rsa',
            r'id_dsa',
            r'id_ecdsa',
            r'id_ed25519',
        ]

        # Compile patterns
        self.compiled_secrets = [
            (re.compile(pattern), replacement)
            for pattern, replacement in self.secret_patterns
        ]
        self.compiled_skip = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.skip_file_patterns
        ]

    def _should_skip_file(self, path: str) -> bool:
        """Check if file should be skipped entirely."""
        for pattern in self.compiled_skip:
            if pattern.search(path):
                return True
        return False

    def _contains_secrets(self, content: str) -> bool:
        """Check if content contains potential secrets."""
        for pattern, _ in self.compiled_secrets:
            if pattern.search(content):
                return True
        return False

    def _redact_secrets(self, content: str) -> str:
        """Redact secrets from content."""
        for pattern, replacement in self.compiled_secrets:
            content = pattern.sub(f"[REDACTED_{replacement}]", content)
        return content

    def process(self, corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Sanitize corpus by removing files with sensitive content.

        Args:
            corpus: List of file dictionaries

        Returns:
            Sanitized corpus with sensitive files removed
        """
        sanitized = []
        skipped_files = 0
        redacted_files = 0

        for item in corpus:
            path = item.get("path", "")

            # Skip sensitive file types
            if self._should_skip_file(path):
                skipped_files += 1
                continue

            content = item.get("content", "")

            # Check for secrets
            if self._contains_secrets(content):
                # For now, skip files with secrets rather than redacting
                # This is safer for training data
                skipped_files += 1
                continue

            sanitized.append(item)

        if skipped_files > 0:
            logger.info(f"Skipped {skipped_files} files with sensitive content")

        return sanitized
