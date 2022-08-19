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
    link = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getContentsByName?name=TCIA_TCGA-PRAD_08-09-2016-v3"
    get_metadata(link)


def get_metadata(name:str) -> pd.DataFrame:
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
    base_link = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getContentsByName?name="
    full_link = base_link + name
    response = requests.get(full_link)
    assert response.status_code == 200, "Authorization Error: {}".format(response.status_code)
    list__key_data = response.json()


if __name__ == "__main__":
    main()
