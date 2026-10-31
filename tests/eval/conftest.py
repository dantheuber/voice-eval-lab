import pytest

from voicelab.store.artifacts import ArtifactStore


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path)
