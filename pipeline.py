"""
---
Data Pipeline
---

    Algorithms used to process data before modeling.

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

def main():
    link = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getPatient"
    get_metadata(link)


def get_metadata(link:str) -> pd.DataFrame:
    """
    ---
    Get Data
    ---

        Extracts metadata using NBIA api.

    ...

    Uses the requests module to get the metadata
    using the Cancer Imaging Archive's NBIA api
    to extract the metadata into a pandas DataFrame.
    """
    response = requests.get(link)
    print(response.status_code)


if __name__ == "__main__":
    main()
