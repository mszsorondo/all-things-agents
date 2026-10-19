"""
Benchmark Problems for Multi-Agent System Testing
A collection of Python programming tasks of varying complexity
"""

from typing import List, Dict, Any

class BenchmarkProblem:
    """Represents a single benchmark problem"""

    def __init__(self, name: str, description: str, difficulty: str, category: str,
                 expected_functions: List[str] = None, test_cases: List[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.difficulty = difficulty  # easy, medium, hard
        self.category = category
        self.expected_functions = expected_functions or []
        self.test_cases = test_cases or []

    def to_prompt(self) -> str:
        """Convert to a prompt for the agent system"""
        return self.description

# Define benchmark problems
BENCHMARK_PROBLEMS = [
    # Easy Problems
    BenchmarkProblem(
        name="calculator",
        description="""Create a calculator with add, subtract, multiply, and divide functions.
        Include error handling for division by zero and invalid inputs.""",
        difficulty="easy",
        category="math",
        expected_functions=["add", "subtract", "multiply", "divide"],
        test_cases=[
            {"input": "add(2, 3)", "expected": 5.0},
            {"input": "divide(10, 0)", "expected": "Error"},
            {"input": "multiply('a', 3)", "expected": "Error"}
        ]
    ),

    BenchmarkProblem(
        name="string_utils",
        description="""Create a string utility module with the following functions:
        1. reverse_string(s) - reverses a string
        2. count_vowels(s) - counts vowels in a string (a, e, i, o, u)
        3. is_palindrome(s) - checks if a string is a palindrome (ignoring case and spaces)
        4. capitalize_words(s) - capitalizes the first letter of each word
        Include proper error handling for non-string inputs.""",
        difficulty="easy",
        category="string_manipulation",
        expected_functions=["reverse_string", "count_vowels", "is_palindrome", "capitalize_words"]
    ),

    BenchmarkProblem(
        name="temperature_converter",
        description="""Create a temperature converter with functions to convert between Celsius, Fahrenheit, and Kelvin:
        1. celsius_to_fahrenheit(celsius)
        2. fahrenheit_to_celsius(fahrenheit)
        3. celsius_to_kelvin(celsius)
        4. kelvin_to_celsius(kelvin)
        Include validation that temperatures are not below absolute zero (-273.15°C, -459.67°F, 0K).
        Return appropriate error messages for invalid inputs.""",
        difficulty="easy",
        category="conversion",
        expected_functions=["celsius_to_fahrenheit", "fahrenheit_to_celsius", "celsius_to_kelvin", "kelvin_to_celsius"]
    ),

    # Medium Problems
    BenchmarkProblem(
        name="csv_processor",
        description="""Create a CSV data processor with the following functions:
        1. read_csv(filepath) - reads a CSV file and returns a list of dictionaries
        2. write_csv(data, filepath) - writes a list of dictionaries to a CSV file
        3. filter_rows(data, column, value) - filters rows where column equals value
        4. aggregate_column(data, column, operation) - performs sum/avg/min/max on a numeric column
        5. sort_by_column(data, column, ascending=True) - sorts data by specified column
        Include comprehensive error handling for file operations and data validation.""",
        difficulty="medium",
        category="data_processing",
        expected_functions=["read_csv", "write_csv", "filter_rows", "aggregate_column", "sort_by_column"]
    ),

    BenchmarkProblem(
        name="password_validator",
        description="""Create a password validation and strength checker module with:
        1. validate_password(password) - checks if password meets requirements (min 8 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special char)
        2. check_strength(password) - returns strength score (weak/medium/strong/very strong) based on length and complexity
        3. generate_password(length, include_special=True) - generates a random secure password
        4. check_common_password(password) - checks against a list of common passwords (you can hardcode a small list)
        5. suggest_improvements(password) - returns list of suggestions to improve the password
        Include proper error messages and edge case handling.""",
        difficulty="medium",
        category="security",
        expected_functions=["validate_password", "check_strength", "generate_password", "check_common_password", "suggest_improvements"]
    ),

    BenchmarkProblem(
        name="file_organizer",
        description="""Create a file organization utility with these functions:
        1. organize_by_extension(source_dir, target_dir) - organizes files into folders by extension
        2. find_duplicates(directory) - finds duplicate files based on content (use hash)
        3. get_folder_size(directory) - calculates total size of a folder in bytes/KB/MB
        4. rename_files_batch(directory, pattern, replacement) - batch rename files using pattern
        5. cleanup_empty_folders(directory) - removes all empty subdirectories
        Include error handling for file permissions, missing directories, and invalid paths.""",
        difficulty="medium",
        category="file_management",
        expected_functions=["organize_by_extension", "find_duplicates", "get_folder_size", "rename_files_batch", "cleanup_empty_folders"]
    ),

    # Hard Problems
    BenchmarkProblem(
        name="json_api_client",
        description="""Create a generic JSON API client with the following features:
        1. get_request(url, params=None, headers=None) - performs GET request
        2. post_request(url, data=None, headers=None) - performs POST request
        3. put_request(url, data=None, headers=None) - performs PUT request
        4. delete_request(url, headers=None) - performs DELETE request
        5. handle_pagination(url, page_param='page') - automatically handles paginated responses
        6. retry_with_backoff(func, max_retries=3) - implements exponential backoff for failed requests
        Include proper error handling for network errors, timeout, invalid JSON, and HTTP error codes.
        Use the requests library if available, otherwise use urllib.""",
        difficulty="hard",
        category="networking",
        expected_functions=["get_request", "post_request", "put_request", "delete_request", "handle_pagination", "retry_with_backoff"]
    ),

    BenchmarkProblem(
        name="sorting_algorithms",
        description="""Implement multiple sorting algorithms and a performance comparator:
        1. bubble_sort(arr) - implements bubble sort
        2. quick_sort(arr) - implements quicksort
        3. merge_sort(arr) - implements merge sort
        4. insertion_sort(arr) - implements insertion sort
        5. compare_performance(arr_size) - compares performance of all algorithms on random array
        6. is_sorted(arr, ascending=True) - checks if array is sorted
        All sorting functions should handle lists of numbers and strings. Include error handling for invalid inputs.
        The compare_performance function should return execution times for each algorithm.""",
        difficulty="hard",
        category="algorithms",
        expected_functions=["bubble_sort", "quick_sort", "merge_sort", "insertion_sort", "compare_performance", "is_sorted"]
    ),

    BenchmarkProblem(
        name="cache_decorator",
        description="""Create an advanced caching system with decorators:
        1. simple_cache(func) - decorator that caches function results
        2. lru_cache(maxsize=128) - decorator implementing LRU (Least Recently Used) cache
        3. timed_cache(ttl_seconds=60) - decorator that expires cached results after TTL
        4. clear_cache(func) - clears cache for a specific function
        5. get_cache_info(func) - returns cache statistics (hits, misses, size)
        The cache should handle functions with different argument types (including keyword arguments).
        Include proper thread safety if possible, and handle unhashable arguments gracefully.""",
        difficulty="hard",
        category="advanced_python",
        expected_functions=["simple_cache", "lru_cache", "timed_cache", "clear_cache", "get_cache_info"]
    ),

    BenchmarkProblem(
        name="markdown_parser",
        description="""Create a simple Markdown to HTML parser that handles:
        1. parse_headers(text) - converts # headers to <h1>-<h6> tags
        2. parse_emphasis(text) - converts *italic* to <em> and **bold** to <strong>
        3. parse_links(text) - converts [text](url) to <a href="url">text</a>
        4. parse_code(text) - converts `code` to <code> and ``` blocks to <pre><code>
        5. parse_lists(text) - converts - items to <ul><li> and 1. items to <ol><li>
        6. markdown_to_html(text) - combines all parsers to convert full markdown to HTML
        Include error handling for malformed markdown syntax.""",
        difficulty="hard",
        category="text_processing",
        expected_functions=["parse_headers", "parse_emphasis", "parse_links", "parse_code", "parse_lists", "markdown_to_html"]
    )
]

def get_problem_by_name(name: str) -> BenchmarkProblem:
    """Get a specific problem by name"""
    for problem in BENCHMARK_PROBLEMS:
        if problem.name == name:
            return problem
    return None

def get_problems_by_difficulty(difficulty: str) -> List[BenchmarkProblem]:
    """Get all problems of a specific difficulty"""
    return [p for p in BENCHMARK_PROBLEMS if p.difficulty == difficulty]

def get_problems_by_category(category: str) -> List[BenchmarkProblem]:
    """Get all problems in a specific category"""
    return [p for p in BENCHMARK_PROBLEMS if p.category == category]

def get_all_problems() -> List[BenchmarkProblem]:
    """Get all benchmark problems"""
    return BENCHMARK_PROBLEMS

def get_problem_categories() -> List[str]:
    """Get all unique problem categories"""
    return list(set(p.category for p in BENCHMARK_PROBLEMS))

def get_problem_summary() -> Dict[str, Any]:
    """Get a summary of all problems"""
    summary = {
        "total": len(BENCHMARK_PROBLEMS),
        "by_difficulty": {},
        "by_category": {},
        "problems": []
    }

    for difficulty in ["easy", "medium", "hard"]:
        problems = get_problems_by_difficulty(difficulty)
        summary["by_difficulty"][difficulty] = len(problems)

    for category in get_problem_categories():
        problems = get_problems_by_category(category)
        summary["by_category"][category] = len(problems)

    for problem in BENCHMARK_PROBLEMS:
        summary["problems"].append({
            "name": problem.name,
            "difficulty": problem.difficulty,
            "category": problem.category,
            "expected_functions": problem.expected_functions
        })

    return summary

if __name__ == "__main__":
    # Print summary of benchmark problems
    summary = get_problem_summary()
    print(f"📊 Benchmark Problems Summary")
    print(f"Total problems: {summary['total']}")
    print(f"\nBy Difficulty:")
    for diff, count in summary["by_difficulty"].items():
        print(f"  {diff}: {count}")
    print(f"\nBy Category:")
    for cat, count in summary["by_category"].items():
        print(f"  {cat}: {count}")
    print(f"\nProblems:")
    for prob in summary["problems"]:
        print(f"  - {prob['name']} ({prob['difficulty']}, {prob['category']})")
        print(f"    Functions: {', '.join(prob['expected_functions'])}")