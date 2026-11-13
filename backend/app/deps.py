"""FastAPI dependencies: tenant header and shared services."""

import re
from typing import Annotated

from fastapi import Depends, Header, Request

from .config import Settings, get_settings
from .errors import MissingTenantError
from .services.embeddings import EmbeddingService
from .services.llm import ChatService

_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def get_tenant_id(
    x_tenant_id: Annotated[
        str | None,
        Header(alias="X-Tenant-Id", description="Tenant (company / department), e.g. `company-a`.",
               examples=["company-a"]),
    ] = None,
) -> str:
    if x_tenant_id is None or not x_tenant_id.strip():
        raise MissingTenantError("Missing X-Tenant-Id header.")
    tenant = x_tenant_id.strip()
    if not _TENANT_RE.match(tenant):
        raise MissingTenantError(
            "Invalid X-Tenant-Id: use 1-64 letters, digits, '-', '_' or '.', starting with a letter or digit."
        )
    return tenant


def get_embedder(request: Request) -> EmbeddingService:
    return request.app.state.embedder


def get_chat(request: Request) -> ChatService:
    return request.app.state.chat


TenantId = Annotated[str, Depends(get_tenant_id)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[EmbeddingService, Depends(get_embedder)]
ChatDep = Annotated[ChatService, Depends(get_chat)]
