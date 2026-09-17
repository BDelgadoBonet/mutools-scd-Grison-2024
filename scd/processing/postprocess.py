"""Functions to postprocess the results of the model."""

import pickle as pkl


def save_results(output_datafile, dictionary_result, neural_data=None):
    """
    Save the dictionary_result to the output_datafile.

    Args:
        output_datafile (Path): The path to the output data file.
        dictionary_result (dict): The dictionary to be saved.
        neural_data (optional): The neural data to be saved together with the dictionary.
    """

    # Add neural data to dictionary, if any
    if neural_data is not None:
        dictionary_result["data"] = neural_data

    # Check if directory exists, create if not
    if not output_datafile.parent.exists():
        output_datafile.parent.mkdir(parents=True)
    try:
        with open(output_datafile, "wb") as f:
            pkl.dump(dictionary_result, f)
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
