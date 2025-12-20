"""
Tests for GitHub repository collection functionality.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.data.github.repo_filtering import RepoFilter, RepoMetrics, is_permissive_license


class TestRepoFilter:
    """Tests for RepoFilter class."""

    def test_filter_by_stars(self):
        """Test filtering by minimum stars."""
        filter = RepoFilter(min_stars=100)

        # Create mock repos
        high_stars = MagicMock()
        high_stars.stargazers_count = 500
        high_stars.license = MagicMock(key="mit")
        high_stars.size = 1000
        high_stars.archived = False
        high_stars.fork = False

        low_stars = MagicMock()
        low_stars.stargazers_count = 50
        low_stars.license = MagicMock(key="mit")
        low_stars.size = 1000
        low_stars.archived = False
        low_stars.fork = False

        assert filter.should_include(high_stars)
        assert not filter.should_include(low_stars)

    def test_filter_by_license(self):
        """Test filtering by allowed licenses."""
        filter = RepoFilter(allowed_licenses=["mit", "apache-2.0"])

        # MIT licensed repo
        mit_repo = MagicMock()
        mit_repo.stargazers_count = 100
        mit_repo.license = MagicMock(key="mit")
        mit_repo.size = 1000
        mit_repo.archived = False
        mit_repo.fork = False

        # GPL licensed repo
        gpl_repo = MagicMock()
        gpl_repo.stargazers_count = 100
        gpl_repo.license = MagicMock(key="gpl-3.0")
        gpl_repo.size = 1000
        gpl_repo.archived = False
        gpl_repo.fork = False

        # No license
        no_license = MagicMock()
        no_license.stargazers_count = 100
        no_license.license = None
        no_license.size = 1000
        no_license.archived = False
        no_license.fork = False

        assert filter.should_include(mit_repo)
        assert not filter.should_include(gpl_repo)
        assert not filter.should_include(no_license)

    def test_filter_by_size(self):
        """Test filtering by maximum size."""
        filter = RepoFilter(max_size_mb=10)  # 10MB max

        # Small repo
        small_repo = MagicMock()
        small_repo.stargazers_count = 100
        small_repo.license = MagicMock(key="mit")
        small_repo.size = 5000  # 5MB
        small_repo.archived = False
        small_repo.fork = False

        # Large repo
        large_repo = MagicMock()
        large_repo.stargazers_count = 100
        large_repo.license = MagicMock(key="mit")
        large_repo.size = 15000  # 15MB
        large_repo.archived = False
        large_repo.fork = False

        assert filter.should_include(small_repo)
        assert not filter.should_include(large_repo)

    def test_filter_archived_repos(self):
        """Test that archived repos are filtered out."""
        filter = RepoFilter()

        archived = MagicMock()
        archived.stargazers_count = 100
        archived.license = MagicMock(key="mit")
        archived.size = 1000
        archived.archived = True
        archived.fork = False

        active = MagicMock()
        active.stargazers_count = 100
        active.license = MagicMock(key="mit")
        active.size = 1000
        active.archived = False
        active.fork = False

        assert not filter.should_include(archived)
        assert filter.should_include(active)

    def test_filter_forks(self):
        """Test that forks are filtered out."""
        filter = RepoFilter()

        fork = MagicMock()
        fork.stargazers_count = 100
        fork.license = MagicMock(key="mit")
        fork.size = 1000
        fork.archived = False
        fork.fork = True

        original = MagicMock()
        original.stargazers_count = 100
        original.license = MagicMock(key="mit")
        original.size = 1000
        original.archived = False
        original.fork = False

        assert not filter.should_include(fork)
        assert filter.should_include(original)

    def test_filter_repos_list(self):
        """Test filtering a list of repos."""
        filter = RepoFilter(min_stars=100)

        repos = []
        for stars in [50, 100, 200, 30, 500]:
            repo = MagicMock()
            repo.stargazers_count = stars
            repo.license = MagicMock(key="mit")
            repo.size = 1000
            repo.archived = False
            repo.fork = False
            repo.full_name = f"user/repo-{stars}"
            repos.append(repo)

        filtered = filter.filter_repos(repos)

        assert len(filtered) == 3  # 100, 200, 500 stars

    def test_get_metrics(self):
        """Test extracting metrics from a repo."""
        filter = RepoFilter()

        repo = MagicMock()
        repo.stargazers_count = 500
        repo.forks_count = 50
        repo.open_issues_count = 10
        repo.license = MagicMock(key="mit")
        repo.language = "TypeScript"
        repo.size = 5000
        repo.updated_at = "2024-01-01"

        metrics = filter.get_metrics(repo)

        assert isinstance(metrics, RepoMetrics)
        assert metrics.stars == 500
        assert metrics.forks == 50
        assert metrics.license_key == "mit"
        assert metrics.language == "TypeScript"


class TestLicenseChecker:
    """Tests for license checking functionality."""

    def test_permissive_licenses(self):
        """Test identification of permissive licenses."""
        permissive = ["mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"]

        for license in permissive:
            assert is_permissive_license(license), f"{license} should be permissive"

    def test_non_permissive_licenses(self):
        """Test identification of non-permissive licenses."""
        copyleft = ["gpl-2.0", "gpl-3.0", "agpl-3.0", "lgpl-3.0"]

        for license in copyleft:
            assert not is_permissive_license(license), f"{license} should not be permissive"

    def test_case_insensitive(self):
        """Test that license check is case insensitive."""
        assert is_permissive_license("MIT")
        assert is_permissive_license("Apache-2.0")
        assert is_permissive_license("BSD-3-Clause")


class TestGitHubCollectorIntegration:
    """Integration tests for GitHubCollector (requires mocking)."""

    @patch.dict("os.environ", {"GITHUB_PAT": "fake_token"})
    @patch("src.data.github.github_client.Github")
    def test_collector_initialization(self, mock_github):
        """Test collector initialization with mocked GitHub client."""
        from src.data.github.github_client import GitHubCollector

        mock_github_instance = MagicMock()
        mock_github_instance.get_rate_limit.return_value = MagicMock(
            core=MagicMock(remaining=5000, limit=5000)
        )
        mock_github.return_value = mock_github_instance

        collector = GitHubCollector()

        assert collector.languages == ["TypeScript", "JavaScript"]
        assert collector.min_stars == 100

    @patch.dict("os.environ", {"GITHUB_PAT": ""})
    def test_collector_requires_token(self):
        """Test that collector requires GitHub token."""
        from src.data.github.github_client import GitHubCollector

        with pytest.raises(ValueError, match="GitHub token required"):
            GitHubCollector()
