import os

import pandas as pd

from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        # validate dataset columns match with schema columns
        try:
            validation_status = True  # Initialize as True

            data = pd.read_csv(self.config.unzip_data_dir)
            data_cols = list(data.columns)

            schema_cols = (
                self.config.all_schema.keys()
            )  # since we use ConfigBox...we can call .keys
            schema_dtype = self.config.all_schema.values()

            for col in data_cols:
                if col not in schema_cols:
                    validation_status = False
                    logger.error("Mismatch dataset column name with the schema.yaml")
                    break  # No need to continue if a mismatch is found

                # Check data type if column is in schema
                expected_type = list(schema_dtype)
                # actual_type = data.dtypes.astype(str).tolist()
                actual_type = list(data.dtypes.astype(str))

                if actual_type != expected_type:
                    validation_status = False
                    logger.error(
                        "Mismatch dataset dtype name with the data type in schema.yaml"
                    )
                    break  # Mismatch found, exit loop

            # Write validation status to file after the loop completes
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            # Handle exceptions accordingly (e.g., log the error)
            logger.error(f"Validation failed due to error: {e}")
            print(f"Validation failed due to error: {e}")
            return False  # Return False on exception
