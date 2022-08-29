SELECT
	dicom_pivot_v10.PatientID,
	dicom_pivot_v10.collection_id,
	dicom_pivot_v10.source_DOI,
	dicom_pivot_v10.StudyInstanceUID,
	dicom_pivot_v10.SeriesInstanceUID,
	dicom_pivot_v10.SOPInstanceUID,
	dicom_pivot_v10.gcs_url
FROM
	`bigquery-public-data.idc_v10.dicom_pivot_v10` dicom_pivot_v10
WHERE
	StudyInstanceUID IN (
		SELECT
			StudyInstanceUID
		FROM
			`bigquery-public-data.idc_v10.dicom_pivot_v10` dicom_pivot_v10
		WHERE
			StudyInstanceUID IN (
				SELECT
					StudyInstanceUID
				FROM
					`bigquery-public-data.idc_v10.dicom_pivot_v10` dicom_pivot_v10
				WHERE
        			(
        				LOWER(dicom_pivot_v10.tcia_tumorLocation) LIKE LOWER('Breast')
        			)
        			GROUP BY
        				StudyInstanceUID
        			INTERSECT DISTINCT
        			SELECT
        				StudyInstanceUID
        			FROM
        				`bigquery-public-data.idc_v10.dicom_pivot_v10` dicom_pivot_v10
        			WHERE
        				(LOWER(dicom_pivot_v10.access) LIKE LOWER('Public'))
        			GROUP BY
        				StudyInstanceUID
        			INTERSECT DISTINCT
        			SELECT
        				StudyInstanceUID
        			FROM
        				`bigquery-public-data.idc_v10.dicom_pivot_v10` dicom_pivot_v10
        			WHERE
				(
					LOWER(dicom_pivot_v10.SOPClassUID) LIKE LOWER('1.2.840.10008.5.1.4.1.1.13.1.3')
				)
        			GROUP BY
        				StudyInstanceUID
			)
			GROUP BY
				StudyInstanceUID
	)
GROUP BY
	dicom_pivot_v10.PatientID,
	dicom_pivot_v10.collection_id,
	dicom_pivot_v10.source_DOI,
	dicom_pivot_v10.StudyInstanceUID,
	dicom_pivot_v10.SeriesInstanceUID,
	dicom_pivot_v10.SOPInstanceUID,
	dicom_pivot_v10.gcs_url
ORDER BY
	dicom_pivot_v10.PatientID ASC,
	dicom_pivot_v10.collection_id ASC,
	dicom_pivot_v10.source_DOI ASC,
	dicom_pivot_v10.StudyInstanceUID ASC,
	dicom_pivot_v10.SeriesInstanceUID ASC,
	dicom_pivot_v10.SOPInstanceUID ASC,
	dicom_pivot_v10.gcs_url ASC
