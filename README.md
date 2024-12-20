# Bob - Your 24/7 Discord Assistant ( IN PROGRESS / NOT WORKING YET )

## Overview

Bob is an intelligent virtual assistant designed to be an active member of any Discord community around the clock. Powered by advanced AI and integrated seamlessly into the Discord environment, Bob aims to enhance user interactions, provide support, and facilitate various tasks within the Discord ecosystem.

## Features

- **Continuous Presence:** Bob remains online 24/7 in the Discord voice and text channels, ensuring constant availability for assistance and interaction.
- **Audio Interaction:** Capable of listening to ongoing conversations and responding when prompted, Bob can engage in meaningful dialogues based on his extensive knowledge base.
- **Knowledge Systems:** Bob's intelligence is powered by multiple memory systems, including:
  - Personal Memories
  - World Knowledge
  - Social Knowledge
  - Rules of Engagement
  - Various types of memories (Episodic, Semantic, Procedural, Implicit, Metacognition, Causal Models, etc.)
- **Virtual Tool Control:** Bob can navigate and control virtual applications such as Discord, Google Chrome, Notepad, and Visual Studio Code to perform tasks and manage projects.
- **AI Project Focus:** Specializes in creating and learning from AI projects, particularly utilizing the [Swarms library](https://github.com/kyegomez/swarms).
- **Mentorship Interaction:** Interacts with mentors like Peyton to continuously refine his knowledge and capabilities.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker installed and running
- Discord account and server where Bob will operate
- OpenAI API Key

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/peytontoblert/bob-agi.git
   cd bob-agi
   ```

2. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Docker Containers:**
   Ensure Docker is running, then launch necessary containers:
   ```bash
   docker-compose up -d
   ```

### Configuration

- **Discord Bot Setup:**
  - Create a new Discord application and bot.
  - Invite the bot to your Agora server with appropriate permissions.
  - Update the Discord token in the `.env` file.

- **Customize Settings:**
  Modify configuration files in the `app/env/` directory to tailor Bob's behavior and integrations as needed.

## Usage

Run the main script to start Bob:
```bash
python main.py
```

## Testing

To run the tests, use the following command:
   ```bash
   python tests/run_tests.py
```
### Using pytest directly:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories using markers
pytest -m unit  # Run unit tests only
pytest -m integration  # Run integration tests only
pytest -m "not slow"  # Skip slow tests

# Run specific test files or directories
pytest tests/unit/  # Run all unit tests
pytest tests/unit/test_browser.py  # Run specific test file

# Run with different verbosity levels
pytest -v  # verbose
pytest -vv  # more verbose
```

### Using the run_tests.py script:
```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --markers unit
python tests/run_tests.py --markers "!slow"  # Skip slow tests
python tests/run_tests.py --markers "unit" "!slow"  # Run unit tests but skip slow ones

# Run specific path
python tests/run_tests.py --path tests/unit/test_browser.py

# Increase verbosity
python tests/run_tests.py -v
python tests/run_tests.py -vv

# Disable coverage
python tests/run_tests.py --no-coverage
```

### Best practices for running tests:
```bash
# 1. Run unit tests during development (fast feedback)
pytest -m unit

# 2. Run with coverage before committing
pytest --cov=app --cov-report=html

# 3. Run full test suite before pushing
pytest

# 4. Run specific tests when debugging
pytest tests/path/to/test.py -vv -s  # -s shows print statements

# 5. Run tests matching a specific name pattern
pytest -k "test_browser"  # Runs all tests with "test_browser" in the name
```


### Example of a unit test:

```bash
pytest tests/unit/computer/test_screen.py -v
```

Bob will initialize his environment, join the Discord voice channel, and begin interacting based on the defined parameters.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---