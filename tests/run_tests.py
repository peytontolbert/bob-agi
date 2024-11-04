import pytest
import sys
import argparse
from pathlib import Path

def run_tests(test_path=None, pattern="test_*.py", markers=None, verbose=2, coverage=True):
    """Run tests with specified parameters
    
    Args:
        test_path (str, optional): Specific test path to run
        pattern (str, optional): Test file pattern to match
        markers (list, optional): List of pytest markers to run/skip
        verbose (int, optional): Verbosity level (0-3)
        coverage (bool, optional): Whether to run coverage report
    """
    args = ["-v" * verbose]
    
    # Add coverage options if requested
    if coverage:
        args.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    # Add test path or use default
    if test_path:
        args.append(str(Path(test_path)))
    else:
        args.append(str(Path(__file__).parent))
    
    # Add markers if specified
    if markers:
        for marker in markers:
            if marker.startswith('!'):
                args.append(f"-m not {marker[1:]}")
            else:
                args.append(f"-m {marker}")
    
    # Run pytest with constructed arguments
    return pytest.main(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the test suite")
    parser.add_argument("--path", help="Specific test path to run")
    parser.add_argument("--pattern", default="test_*.py", help="Test file pattern")
    parser.add_argument("--markers", nargs="+", help="Pytest markers to run/skip (prefix with ! to skip)")
    parser.add_argument("--verbose", "-v", action="count", default=1, help="Increase verbosity")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    
    args = parser.parse_args()
    success = run_tests(
        args.path, 
        args.pattern, 
        args.markers, 
        args.verbose + 1,
        not args.no_coverage
    )
    sys.exit(0 if success == 0 else 1) 