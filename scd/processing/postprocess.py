"""Functions to postprocess the results of the model."""

import pickle as pkl
from pathlib import Path


def save_results(output_datafile: Path | str, dictionary_result: dict, neural_data = None):
    """
    Save the dictionary_result to the output_datafile.

    Args:
        output_datafile (Path | str): The path to the output data file.
        dictionary_result (dict): The dictionary to be saved.
        neural_data (Tensor | None): The neural data to be saved together with the dictionary (optional).
    """

    # Add neural data to dictionary, if any
    if neural_data is not None:
        dictionary_result["data"] = neural_data

    # Check if directory exists, create if not
    output_path = Path(output_datafile)  # accept strings
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
    try:
        with open(output_path, "wb") as f:
            pkl.dump(dictionary_result, f)
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
