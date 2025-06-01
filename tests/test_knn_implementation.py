"""
Unit tests for advanced KNN implementation.
"""

import unittest
from unittest.mock import patch
import numpy as np


# The module uses try/except for imports, so we need to mock them
@patch("proctor.few_shot.knn_implementation._DEPS_AVAILABLE", True)
@patch("proctor.few_shot.knn_implementation.SentenceTransformer")
@patch("proctor.few_shot.knn_implementation.cosine_similarity")
class TestSemanticKNN(unittest.TestCase):
    """Test cases for the SemanticKNN implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here after mocking
        from proctor.few_shot.knn_implementation import SemanticKNN, EmbeddingCache

        self.SemanticKNN = SemanticKNN
        self.EmbeddingCache = EmbeddingCache

    def test_embedding_cache_init(self, mock_cos_sim, mock_transformer):
        """Test EmbeddingCache initialization."""
        cache = self.EmbeddingCache(max_size=100)
        self.assertEqual(cache.max_size, 100)
        self.assertEqual(len(cache.cache), 0)

    def test_embedding_cache_add_get(self, mock_cos_sim, mock_transformer):
        """Test adding and retrieving embeddings from cache."""
        cache = self.EmbeddingCache(max_size=3)

        # Create some test embeddings
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])

        # Add to cache
        cache.add("text1", embedding1)
        cache.add("text2", embedding2)

        # Retrieve from cache
        self.assertTrue(np.array_equal(cache.get("text1"), embedding1))
        self.assertTrue(np.array_equal(cache.get("text2"), embedding2))
        self.assertIsNone(cache.get("nonexistent"))

    def test_embedding_cache_max_size(self, mock_cos_sim, mock_transformer):
        """Test that cache respects max_size and implements LRU behavior."""
        cache = self.EmbeddingCache(max_size=2)

        # Create some test embeddings
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])
        embedding3 = np.array([0.7, 0.8, 0.9])

        # Add to cache
        cache.add("text1", embedding1)
        cache.add("text2", embedding2)
        cache.add("text3", embedding3)  # This should remove text1

        # Check that text1 was removed and text2, text3 remain
        self.assertIsNone(cache.get("text1"))
        self.assertTrue(np.array_equal(cache.get("text2"), embedding2))
        self.assertTrue(np.array_equal(cache.get("text3"), embedding3))

    def test_semantic_knn_init(self, mock_cos_sim, mock_transformer):
        """Test SemanticKNN initialization."""
        knn = self.SemanticKNN(model_name="test-model", cache_size=100)

        # Check that model was initialized with correct parameters
        mock_transformer.assert_called_once_with("test-model")

        # Check that cache was initialized
        self.assertEqual(knn.cache.max_size, 100)

    def test_get_embedding_new(self, mock_cos_sim, mock_transformer):
        """Test getting a new embedding (not in cache)."""
        # Set up the mock model
        mock_model = mock_transformer.return_value
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding

        knn = self.SemanticKNN()
        result = knn._get_embedding("test text")

        # Check that model.encode was called with correct parameters
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

        # Check that result is correct
        self.assertTrue(np.array_equal(result, mock_embedding))

        # Check that embedding was added to cache
        self.assertTrue(np.array_equal(knn.cache.get("test text"), mock_embedding))

    def test_get_embedding_cached(self, mock_cos_sim, mock_transformer):
        """Test getting an embedding from cache."""
        # Set up the mock model
        mock_model = mock_transformer.return_value
        mock_embedding = np.array([0.1, 0.2, 0.3])

        knn = self.SemanticKNN()

        # Add embedding to cache manually
        knn.cache.add("test text", mock_embedding)

        # Get embedding
        result = knn._get_embedding("test text")

        # Check that model.encode was NOT called
        mock_model.encode.assert_not_called()

        # Check that result is correct (from cache)
        self.assertTrue(np.array_equal(result, mock_embedding))

    def test_find_nearest(self, mock_cos_sim, mock_transformer):
        """Test finding nearest neighbors."""
        # Set up mocks
        mock_model = mock_transformer.return_value
        query_embedding = np.array([[0.1, 0.2, 0.3]])
        candidate_embeddings = np.array(
            [
                [0.4, 0.5, 0.6],  # candidate 1
                [0.7, 0.8, 0.9],  # candidate 2
            ]
        )

        # Mock the encode method to return different embeddings for different inputs
        def mock_encode(text, convert_to_numpy):
            if text == "query":
                return query_embedding
            elif text == "candidate1":
                return candidate_embeddings[0]
            elif text == "candidate2":
                return candidate_embeddings[1]

        mock_model.encode.side_effect = mock_encode

        # Mock cosine_similarity to return predetermined values
        mock_cos_sim.return_value = np.array(
            [[0.8, 0.6]]
        )  # candidate1 is closer than candidate2

        knn = self.SemanticKNN()

        candidates = [
            {"input": "candidate1", "output": "output1"},
            {"input": "candidate2", "output": "output2"},
        ]

        results = knn.find_nearest("query", candidates, k=1)

        # Check that cosine_similarity was called with correct parameters
        mock_cos_sim.assert_called_once()

        # Check that results are correct
        self.assertEqual(len(results), 1)  # k=1
        self.assertEqual(
            results[0][0], candidates[0]
        )  # candidate1 has higher similarity
        self.assertEqual(results[0][1], 0.8)  # similarity score

    def test_find_nearest_empty_candidates(self, mock_cos_sim, mock_transformer):
        """Test finding nearest neighbors with empty candidates list."""
        knn = self.SemanticKNN()
        results = knn.find_nearest("query", [], k=3)

        # Should return empty list
        self.assertEqual(results, [])

        # No embeddings should be computed
        mock_transformer.return_value.encode.assert_not_called()

        # No similarity calculations should be performed
        mock_cos_sim.assert_not_called()


# Test with missing dependencies
class TestSemanticKNNMissingDeps(unittest.TestCase):
    """Test cases for the SemanticKNN with missing dependencies."""

    @patch("proctor.few_shot.knn_implementation._DEPS_AVAILABLE", False)
    def test_missing_dependencies(self):
        """Test that ImportError is raised when dependencies are missing."""
        from proctor.few_shot.knn_implementation import SemanticKNN

        with self.assertRaises(ImportError) as context:
            knn = SemanticKNN()

        self.assertIn("Required dependencies not available", str(context.exception))


if __name__ == "__main__":
    unittest.main()
