import numpy as np
import pytest
from brisket import encode_seq


def test_encode_seq_basic():
    """Test basic DNA sequence encoding."""
    seq = "ATCG"
    result = encode_seq(seq)
    
    # Check shape
    assert result.shape == (4, 4)
    assert result.dtype == np.uint8
    
    # Check encoding
    expected = np.array([
        [1, 0, 0, 0],  # A
        [0, 0, 0, 1],  # T
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0]   # G
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(result, expected)


def test_encode_seq_lowercase():
    """Test that lowercase sequences are handled correctly."""
    seq = "atcg"
    result = encode_seq(seq)
    
    expected = np.array([
        [1, 0, 0, 0],  # A
        [0, 0, 0, 1],  # T
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0]   # G
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(result, expected)


def test_encode_seq_mixed_case():
    """Test mixed case input."""
    seq = "AtCg"
    result = encode_seq(seq)
    
    expected = np.array([
        [1, 0, 0, 0],  # A
        [0, 0, 0, 1],  # T
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0]   # G
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(result, expected)


def test_encode_seq_empty():
    """Test empty sequence."""
    seq = ""
    result = encode_seq(seq)
    
    assert result.shape == (0, 4)
    assert result.dtype == np.uint8


def test_encode_seq_single_base():
    """Test single base sequences."""
    for base, expected_pos in [('A', 0), ('C', 1), ('G', 2), ('T', 3)]:
        result = encode_seq(base)
        assert result.shape == (1, 4)
        
        expected = np.zeros((1, 4), dtype=np.uint8)
        expected[0, expected_pos] = 1
        
        np.testing.assert_array_equal(result, expected)


def test_encode_seq_long_sequence():
    """Test longer DNA sequence."""
    seq = "AAATTTCCCGGG"
    result = encode_seq(seq)
    
    assert result.shape == (12, 4)
    
    # Check that A positions have 1 in column 0
    assert np.all(result[0:3, 0] == 1)
    assert np.all(result[0:3, 1:] == 0)
    
    # Check that T positions have 1 in column 3
    assert np.all(result[3:6, 3] == 1)
    assert np.all(result[3:6, 0:3] == 0)
    
    # Check that C positions have 1 in column 1
    assert np.all(result[6:9, 1] == 1)
    assert np.all(result[6:9, [0, 2, 3]] == 0)
    
    # Check that G positions have 1 in column 2
    assert np.all(result[9:12, 2] == 1)
    assert np.all(result[9:12, [0, 1, 3]] == 0)


def test_encode_seq_invalid_characters():
    """Test sequence with invalid characters (should be ignored)."""
    seq = "ATCGN"  # N is not A, T, C, or G
    result = encode_seq(seq)
    
    assert result.shape == (5, 4)
    
    # First 4 bases should be encoded normally
    expected_valid = np.array([
        [1, 0, 0, 0],  # A
        [0, 0, 0, 1],  # T
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0]   # G
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(result[0:4], expected_valid)
    
    # N should result in all zeros
    np.testing.assert_array_equal(result[4], np.zeros(4, dtype=np.uint8))
