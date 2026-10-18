"""Informative exception types used by MedVerify-RL."""


class MedVerifyError(Exception):
    """Base exception for expected application errors."""


class ConfigurationError(MedVerifyError, ValueError):
    """Raised when an environment setting is invalid."""


class SchemaValidationError(MedVerifyError, ValueError):
    """Raised when a domain record violates its schema constraints."""


class SampleDataError(MedVerifyError):
    """Raised when the development sample file cannot be loaded or parsed."""