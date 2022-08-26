"""Algorithms used to process data before modeling.

...

A set of algorithms used to feed in and process
data before used within the model. This will contain
the data extraction from its rawest form and output
the final form of the data set. The main source of
data will be image related from the Cancer Imaging
Archive.
"""
import requests
import pandas as pd
import json
from os.path import exists

def main():
    """Test the new functions."""
    key_data = get_data(name = "hello", option = "collections")
    print(key_data["collections"][0])


def get_data(name:str, option:str) -> list: # This will be deprecated and no longer in use
    """Extract metadata using NBIA api.

    ...

    Uses the requests module to get the metadata
    using the Cancer Imaging Archive's NBIA api
    to extract the metadata into a pandas DataFrame.

    Parameter(s)
    ---

    name:str
        The name of the research data set.

    option:str
        The kind of api call to be made. the following
        are the possible calls that one can make:
            1. collections - Get a list of collections in the current IDC data version.
            2. cohorts - Get the metadata on the user's cohorts.

    Returns
    ---
    key_data:list
        A list of the samples within the data set.
    """
    assert option is not None, "Please select between on of the following two options:\n1. collections\n2. cohorts\n\nFor more information, please view documentation."
    base_link = "https://api.imaging.datacommons.cancer.gov/v1/"
    if option == "collections":
        full_link = base_link + option
    elif option == "cohorts":
        full_link = base_link + option
    else:
        pass
    response = requests.get(full_link)
    assert response.status_code == 200, "Authorization Error: {}".format(response.status_code)
    key_data = response.json()
    if exists('keys.txt') is False:
        with open("keys.txt", 'w') as fp:
            fp.write(str(key_data))
            fp.close()
    else:
        pass
    return key_data


if __name__ == "__main__":
    main()
