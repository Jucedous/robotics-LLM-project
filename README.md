# Project Overview

This project is designed to facilitate querying machine learning models using predefined prompts. It allows users to select prompts, fill them with JSON data representing objects, and retrieve responses from various models.

## Project Structure

- **get_response.py**: Contains the main logic for querying models based on prompts. It includes functions for:
  - Getting installed models
  - Querying models
  - Selecting prompts
  - Filling prompts with JSON data
  - Selecting models
  - Writing responses to files

- **prompt.py**: Defines various prompt templates that can be used in the application. It contains constants for prompts referenced in `get_response.py`.

- **objects/**: This directory holds subdirectories containing JSON files that represent objects and their coordinates. The structure within this directory will depend on the specific objects and JSON files included.

- **response/**: This directory is where the response files will be saved after querying the models. It will be created automatically if it does not exist.

## Setup Instructions

1. Clone the repository or download the project files.
2. Navigate to the project directory.
3. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

## Usage Instructions

1. Ensure that the necessary models are installed and accessible via the command line.
2. Run the main script:
   ```
   python get_response.py
   ```
3. Follow the prompts to select the desired models and prompts for querying.