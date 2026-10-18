"""Expected errors for the live evidence-verification service."""

from src.common.exceptions import MedVerifyError


class VerificationInputError(MedVerifyError, ValueError):
    """Raised when a verification request is invalid or exceeds limits."""


class EvidenceRetrievalError(MedVerifyError):
    """Raised for network, HTTP, or malformed Europe PMC responses."""


class PolicyCompatibilityError(MedVerifyError):
    """Raised when a simulator policy is incorrectly applied to live evidence."""


class AuditWriteError(MedVerifyError):
    """Raised when the required evidence ledger cannot be persisted."""