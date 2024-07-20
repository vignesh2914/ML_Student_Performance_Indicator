import sys
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from src.logger import logging
from dataclasses import dataclass


from src.utils import save_object



@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join('artifacts','preprossor.pkl')

class DataTransformation:
    def __init__(self):

        self.data_transformation_config : DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [

                "gender",
                "race_ethnicity",
                "parent_level_of_education",
                "lunch",
                "test_preparation_course",

            ]

            num_pipeline = Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scalar",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                
                steps =[
                ("imputer", SimpleImputer(strategy="median")),
                ("onehot", OneHotEncoder())
                ("scalar",StandardScaler)
                ]
            )

            logging.info("Numerical coloums standard scaling completed")
            logging.info("categorical coloums standard scaling completed")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(test_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data Read completed")
            
            logging.info("Obtaining Preprocessing object")

            target_column_name= "math_score"
            
            numerical_columns = ["writing_score","reading_score"]

            #Train Dataframe

            input_feature_train_df =train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            #Test Dataframe

            input_feature_test_df =train_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = train_df[target_column_name]


            logging.info(f"Applying Preprocessing object on Training df and testing df")


            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #fit_transform
            #Calculates the necessary statistics (e.g., mean, variance for scaling) 
            # from the training data and then applies the transformation.

            # transform
            # Uses the already learned parameters (from the training data) to transform the test or validation data.
            train_arr = np.c_
            [
            
            input_feature_train_arr, np.array(target_feature_train_df)

            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # c denoted Function: Concatenates arrays along columns (axis=1).
            # Useful when you want to combine feature arrays and target arrays into a single dataset.


            logging.info(f"Saved preprocessing object.")
            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

            
            













    






