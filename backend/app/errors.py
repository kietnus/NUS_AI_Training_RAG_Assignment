"""Application errors. Each maps to an HTTP status and a clear, human-readable message."""


class AppError(Exception):
    status_code = 400
    code = "bad_request"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class MissingTenantError(AppError):
    status_code = 400
    code = "missing_tenant"


class MetadataError(AppError):
    """Required document metadata is missing or invalid. The document is NOT indexed."""

    status_code = 422
    code = "invalid_metadata"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "unsupported_file_type"


class FileTooLargeError(AppError):
    status_code = 413
    code = "file_too_large"


class ParseError(AppError):
    status_code = 422
    code = "parse_error"


class DuplicateError(AppError):
    status_code = 409
    code = "duplicate"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ModelServiceError(AppError):
    """Ollama (embedding / chat model) is unreachable or misconfigured."""

    status_code = 503
    code = "model_service_unavailable"
