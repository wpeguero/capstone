SELECT
    dicom_pivot_v11.PatientID,
    dicom_pivot_v11.collection_id,
    dicom_pivot_v11.source_DOI,
    dicom_pivot_v11.StudyInstanceUID,
    dicom_pivot_v11.SeriesInstanceUID,
    dicom_pivot_v11.SOPInstanceUID,
    dicom_pivot_v11.gcs_url
FROM
    `bigquery-public-data.idc_v11.dicom_pivot_v11` dicom_pivot_v11
WHERE
    StudyInstanceUID IN (
        SELECT
            StudyInstanceUID
        FROM
            `bigquery-public-data.idc_v11.dicom_pivot_v11` dicom_pivot_v11
        WHERE
            StudyInstanceUID IN (
                SELECT
                StudyInstanceUID
                FROM
                    `bigquery-public-data.idc_v11.dicom_pivot_v11` dicom_pivot_v11
                WHERE
                (
                    LOWER(dicom_pivot_v11.tcia_tumorLocation) LIKE LOWER('Breast')
                )
                GROUP BY
                    StudyInstanceUID
                INTERSECT DISTINCT
                SELECT
                    StudyInstanceUID
                FROM
                    `bigquery-public-data.idc_v11.dicom_pivot_v11` dicom_pivot_v11
                WHERE
                    (LOWER(dicom_pivot_v11.access) LIKE LOWER('Public'))
                GROUP BY
                    StudyInstanceUID
            )
        GROUP BY
            StudyInstanceUID
    )
GROUP BY
    dicom_pivot_v11.PatientID,
    dicom_pivot_v11.collection_id,
    dicom_pivot_v11.source_DOI,
    dicom_pivot_v11.StudyInstanceUID,
    dicom_pivot_v11.SeriesInstanceUID,
    dicom_pivot_v11.SOPInstanceUID,
    dicom_pivot_v11.gcs_url
ORDER BY
    dicom_pivot_v11.PatientID ASC,
    dicom_pivot_v11.collection_id ASC,
    dicom_pivot_v11.source_DOI ASC,
    dicom_pivot_v11.StudyInstanceUID ASC,
    dicom_pivot_v11.SeriesInstanceUID ASC,
    dicom_pivot_v11.SOPInstanceUID ASC,
    dicom_pivot_v11.gcs_url ASC