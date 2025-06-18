import os
import pandas as pd
import openai
from dotenv import load_dotenv
import random

# --- Configuration ---

# Load environment variables from a .env file (for your API key)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
SUBSET_SIZE = 30
# This path assumes you have downloaded the parquet file from
# Hugging Face (princeton-nlp/SWE-bench_Verified) and placed it in a 'data' folder.
DATASET_PATH = "data/test-00000-of-00001.parquet"

# --- Core Agent Logic ---

def load_dataset_subset(path: str, subset_size: int) -> pd.DataFrame:
    """
    Loads the SWE-bench dataset from a parquet file and returns a random subset.

    Args:
        path (str): The file path to the parquet dataset.
        subset_size (int): The number of problems to select.

    Returns:
        pd.DataFrame: A DataFrame containing the subset of problems.
    """
    print(f"Loading dataset from {path}...")
    try:
        df = pd.read_parquet(path)
        print(f"Dataset loaded successfully with {len(df)} total problems.")
        if len(df) < subset_size:
            print(f"Warning: Requested subset size ({subset_size}) is larger than the dataset ({len(df)}). Using all available problems.")
            return df
        # Return a random sample of the specified size
        return df.sample(n=subset_size, random_state=42)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{path}'.")
        print("Please ensure you have downloaded the SWE-bench Verified dataset and placed it correctly.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return pd.DataFrame()

def construct_prompt(problem_statement: str, repo: str) -> str:
    """
    Constructs a detailed prompt for the LLM to generate a code patch.

    Args:
        problem_statement (str): The text from the GitHub issue.
        repo (str): The name of the repository (e.g., 'owner/repo').

    Returns:
        str: A formatted prompt string for the AI model.
    """
    return f"""You are an expert AI software engineer. Your task is to fix a bug in a Python repository.
                You will be given a problem statement from a GitHub issue and you must generate a code patch in the `diff` format to resolve the issue.

                **Repository:** {repo}

                **Problem Statement (from GitHub Issue):**
                ---
                {problem_statement}
                ---

                **Instructions:**
                1.  Carefully analyze the problem statement to understand the bug.
                2.  Reason about the location of the bug in the codebase.
                3.  Generate a code patch that fixes the bug.
                4.  The patch must be in the standard `diff` format, starting with `--- a/path/to/file.py` and `+++ b/path/to/file.py`.
                5.  Only provide the code patch. Do not include any other text, explanations, or comments outside of the patch itself.

                **Generated Patch:**
                """

def get_llm_suggestion(prompt: str) -> str:
    """
    Calls the OpenAI API to get a suggested code patch from the LLM.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The model's generated response (hopefully a diff patch).
    """
    try:
        print("Contacting the AI model for a solution...")
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert AI software engineer specializing in bug fixes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0, # Set to 0 for deterministic and focused output
            # max_tokens=2048,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        suggestion = response.choices[0].message.content
        print("Suggestion received.")
        return suggestion.strip()
    except openai.APIError as e:
        print(f"An OpenAI API error occurred: {e}")
        return f"// Error: Could not get a suggestion from the model. API Error: {e}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"// Error: Could not get a suggestion from the model. Details: {e}"


def run_agent_on_problem(problem: pd.Series):
    """
    Orchestrates the process of running the agent on a single problem.

    Args:
        problem (pd.Series): A row from the DataFrame representing one problem.
    """
    instance_id = problem['instance_id']
    repo = problem['repo']
    problem_statement = problem['problem_statement']

    print("\n" + "="*80)
    print(f"Processing Instance: {instance_id}")
    print(f"Repository: {repo}")
    print("-"*80)
    print("Problem Statement:\n")
    print(problem_statement)
    print("-"*80)

    # 1. Construct the prompt
    prompt = construct_prompt(problem_statement, repo)

    # 2. Get the suggestion from the LLM
    suggested_patch = get_llm_suggestion(prompt)

    # 3. Display the result
    print("\n--- AI Generated Patch ---")
    print(suggested_patch)
    print("="*80 + "\n")


def main():
    print("--- Starting SWE-bench Agent ---")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please create a .env file and add your key.")
        return

    # Load a subset of the dataset
    problem_subset = load_dataset_subset(DATASET_PATH, SUBSET_SIZE)

    if problem_subset.empty:
        print("Could not load dataset. Exiting.")
        return

    print(f"\nSuccessfully selected a subset of {len(problem_subset)} problems.")

    # Iterate through the selected problems and run the agent
    for _, problem in problem_subset.iterrows():
        run_agent_on_problem(problem)

    print("--- Agent has finished processing all problems in the subset. ---")


if __name__ == "__main__":
    main()
