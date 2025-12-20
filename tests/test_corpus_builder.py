"""
Tests for corpus building functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.corpus.pack_jsonl import CorpusPacker, SourceFile
from src.data.corpus.dedupe import Deduplicator, exact_dedupe_by_hash
from src.data.corpus.sanitize import Sanitizer, quick_secret_check
from src.data.corpus.chunking import Chunker, estimate_tokens


class TestCorpusPacker:
    """Tests for CorpusPacker class."""

    def test_detect_language(self):
        """Test language detection from file extension."""
        packer = CorpusPacker()

        assert packer._detect_language(".ts") == "typescript"
        assert packer._detect_language(".tsx") == "typescript"
        assert packer._detect_language(".js") == "javascript"
        assert packer._detect_language(".jsx") == "javascript"
        assert packer._detect_language(".json") == "json"
        assert packer._detect_language(".md") == "markdown"
        assert packer._detect_language(".py") == "unknown"

    def test_extract_files_from_temp_dir(self):
        """Test extracting files from a temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "test.ts").write_text("const x: number = 1;")
            (tmppath / "test.js").write_text("const y = 2;")
            (tmppath / "readme.md").write_text("# Test")

            # Create nested structure
            (tmppath / "src").mkdir()
            (tmppath / "src" / "index.ts").write_text("export const main = () => {};")

            # Create ignored directory
            (tmppath / "node_modules").mkdir()
            (tmppath / "node_modules" / "dep.js").write_text("ignored")

            packer = CorpusPacker()
            files = packer.extract_files(tmppath)

            # Should find 4 files, not including node_modules
            assert len(files) == 4

            # Check file types
            languages = {f["language"] for f in files}
            assert "typescript" in languages
            assert "javascript" in languages
            assert "markdown" in languages

    def test_skip_large_files(self):
        """Test that large files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a file larger than max size
            large_content = "x" * (1024 * 1024 + 1)  # > 1MB
            (tmppath / "large.ts").write_text(large_content)
            (tmppath / "small.ts").write_text("const x = 1;")

            packer = CorpusPacker(max_file_size=1024 * 1024)
            files = packer.extract_files(tmppath)

            assert len(files) == 1
            assert files[0]["path"] == "small.ts"

    def test_save_jsonl(self):
        """Test saving to JSONL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"

            files = [
                {"path": "test.ts", "content": "const x = 1;", "language": "typescript"},
                {"path": "test.js", "content": "const y = 2;", "language": "javascript"},
            ]

            packer = CorpusPacker()
            packer.save_jsonl(files, output_path)

            # Read back and verify
            loaded = []
            with open(output_path) as f:
                for line in f:
                    loaded.append(json.loads(line))

            assert len(loaded) == 2
            assert loaded[0]["path"] == "test.ts"
            assert loaded[1]["language"] == "javascript"


class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_exact_duplicates_removed(self):
        """Test that exact duplicates are removed."""
        docs = [
            {"content": "const x = 1;", "path": "a.ts"},
            {"content": "const x = 1;", "path": "b.ts"},  # Exact duplicate
            {"content": "const y = 2;", "path": "c.ts"},
        ]

        result = exact_dedupe_by_hash(docs)

        assert len(result) == 2

    def test_near_duplicates(self):
        """Test MinHash near-duplicate detection."""
        deduper = Deduplicator(threshold=0.8)

        docs = [
            {"content": "This is a test document with some content.", "path": "a.txt"},
            {"content": "This is a test document with some content!", "path": "b.txt"},  # Very similar
            {"content": "Completely different text here.", "path": "c.txt"},
        ]

        result = deduper.deduplicate(docs)

        # Should keep 2 documents (first and third, second is near-duplicate)
        assert len(result) == 2

    def test_similarity_computation(self):
        """Test similarity score computation."""
        deduper = Deduplicator()

        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy cat"
        text3 = "Completely unrelated text with no overlap"

        sim12 = deduper.compute_similarity(text1, text2)
        sim13 = deduper.compute_similarity(text1, text3)

        assert sim12 > 0.5  # Similar texts
        assert sim13 < 0.3  # Different texts


class TestSanitizer:
    """Tests for Sanitizer class."""

    def test_detect_api_keys(self):
        """Test API key detection."""
        content_with_key = 'const apiKey = "sk_live_abcdefghijklmnopqrstuvwx";'
        content_clean = 'const name = "hello";'

        assert quick_secret_check(content_with_key)
        assert not quick_secret_check(content_clean)

    def test_redact_secrets(self):
        """Test secret redaction."""
        sanitizer = Sanitizer(redact=True)

        doc = {
            "content": 'const token = "ghp_abcdefghijklmnopqrstuvwxyz1234567890";',
            "path": "config.ts",
        }

        result = sanitizer.sanitize_document(doc)

        assert result is not None
        assert "ghp_" not in result["content"]
        assert "[REDACTED" in result["content"]

    def test_skip_env_files(self):
        """Test that .env files are skipped."""
        sanitizer = Sanitizer()

        doc = {
            "content": "API_KEY=secret123",
            "path": ".env.local",
        }

        result = sanitizer.sanitize_document(doc)

        assert result is None

    def test_process_multiple_documents(self):
        """Test processing multiple documents."""
        sanitizer = Sanitizer()

        docs = [
            {"content": "const x = 1;", "path": "clean.ts"},
            {"content": "const key = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890';", "path": "secret.ts"},
            {"content": "", "path": "empty.ts"},
        ]

        result = sanitizer.process(docs)

        # Should have 2 documents (empty filtered, secret redacted)
        assert len(result) == 2


class TestChunker:
    """Tests for Chunker class."""

    def test_small_document_single_chunk(self):
        """Test that small documents result in single chunk."""
        chunker = Chunker(chunk_size=1000)

        doc = {"content": "Small content", "path": "small.ts", "language": "typescript"}
        chunks = list(chunker.chunk_document(doc))

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Small content"
        assert chunks[0]["chunk_index"] == 0

    def test_large_document_multiple_chunks(self):
        """Test that large documents are split into multiple chunks."""
        chunker = Chunker(chunk_size=50, overlap=10)

        # Create content that will definitely need multiple chunks
        long_content = "word " * 200

        doc = {"content": long_content, "path": "large.ts", "language": "typescript"}
        chunks = list(chunker.chunk_document(doc))

        assert len(chunks) > 1

        # Check all chunks have content
        for chunk in chunks:
            assert len(chunk["text"]) > 0

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world, this is a test."

        tokens = estimate_tokens(text)

        # Should be reasonable estimate
        assert 5 <= tokens <= 15


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the full extraction -> sanitize -> dedupe -> chunk pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "clean.ts").write_text("export const add = (a: number, b: number) => a + b;")
            (tmppath / "duplicate.ts").write_text("export const add = (a: number, b: number) => a + b;")
            (tmppath / "secret.ts").write_text('const key = "ghp_abcdefghijklmnopqrstuvwxyz1234567890";')

            # Extract
            packer = CorpusPacker()
            files = packer.extract_files(tmppath)
            assert len(files) == 3

            # Sanitize
            sanitizer = Sanitizer()
            sanitized = sanitizer.process(files)

            # Dedupe
            deduped = exact_dedupe_by_hash(sanitized)

            # Chunk
            chunker = Chunker()
            chunks = chunker.chunk_corpus(deduped)

            assert len(chunks) >= 1
