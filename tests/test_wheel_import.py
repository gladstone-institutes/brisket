"""Test that the wheel installation works correctly."""

import sys
import pytest


def test_brisket_import():
    """Test that brisket can be imported successfully."""
    try:
        import brisket
        assert hasattr(brisket, "encode_seq")
        assert hasattr(brisket, "__version__")
    except ImportError as e:
        pytest.fail(f"Failed to import brisket: {e}")


def test_brisket_version():
    """Test that brisket has a valid version."""
    import brisket
    
    # Version should be a string
    assert isinstance(brisket.__version__, str)
    # Version should not be empty or default
    assert brisket.__version__ != ""
    assert brisket.__version__ != "0.0.0"


def test_encode_seq_function():
    """Test that encode_seq function works after wheel installation."""
    import brisket
    import numpy as np
    
    # Test basic functionality
    result = brisket.encode_seq("ATCG")
    
    # Check basic properties
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
    assert result.dtype == np.uint8
    
    # Check that we get expected encoding
    expected = np.array([
        [1, 0, 0, 0],  # A
        [0, 0, 0, 1],  # T
        [0, 1, 0, 0],  # C
        [0, 0, 1, 0]   # G
    ], dtype=np.uint8)
    
    np.testing.assert_array_equal(result, expected)


def test_cython_extension_loaded():
    """Test that the Cython extension is properly compiled and loaded."""
    import brisket
    
    # Check that we're using the compiled Cython version, not fallback
    try:
        from brisket.brisket import encode_seq as cython_encode_seq
        # If this import succeeds, we have the compiled extension
        assert callable(cython_encode_seq)
    except ImportError:
        pytest.fail("Cython extension not available - using fallback implementation")


def test_performance_indication():
    """Test that the function is reasonably fast (indicating Cython compilation)."""
    import brisket
    import time
    
    # Test with a reasonably long sequence
    long_seq = "ATCG" * 1000  # 4000 base pairs
    
    start_time = time.time()
    result = brisket.encode_seq(long_seq)
    end_time = time.time()
    
    # Should complete quickly (less than 1 second for 4000 bp)
    duration = end_time - start_time
    assert duration < 1.0, f"Function took {duration:.3f}s, may not be using Cython"
    
    # Check result is correct
    assert result.shape == (4000, 4)


def test_numpy_integration():
    """Test that the function integrates properly with numpy."""
    import brisket
    import numpy as np
    
    result = brisket.encode_seq("AAAA")
    
    # Should be a proper numpy array
    assert isinstance(result, np.ndarray)
    assert result.flags.c_contiguous  # Should be C-contiguous
    assert result.flags.owndata  # Should own its data
    
    # Should work with numpy operations
    total = np.sum(result)
    assert total == 4  # 4 bases, each encoded as one 1 in a row of 4