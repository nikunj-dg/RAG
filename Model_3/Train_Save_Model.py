# Databricks notebook source
# MAGIC %md
# MAGIC Train and Save a LLM Model

# COMMAND ----------

# MAGIC %md
# MAGIC Training

# COMMAND ----------

# MAGIC %md
# MAGIC Save the model to Model Registry

# COMMAND ----------

from databricks_langchain import ChatDatabricks

import os
import json
import mlflow
import pandas as pd

# COMMAND ----------

# set the experiment id
mlflow.set_experiment(experiment_id="1927489275036666")

llm = ChatDatabricks(
    endpoint='databricks-dbrx-instruct',
    temperature=0.1,
)

system_prompt = (
  "The following is a conversation with an AI assistant."
  + "The assistant is helpful and very friendly."
)

# start a run
mlflow.start_run()
mlflow.log_param("system_prompt", system_prompt)

# COMMAND ----------

# Define a wrapper for the Databricks-hosted LLM
class DatabricksLLMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, llm):
        self.llm = llm  # Store the ChatDatabricks instance

    def predict(self, model_input):
        questions_list = model_input["question"].tolist()
        responses = self.llm.invoke(questions_list)  # Returns AIMessage objects
        
        # Extract the text content from AIMessage objects
        return [responses.content] 

# Log the model to MLflow
logged_model = mlflow.pyfunc.log_model(
    artifact_path="models:/databricks_llm_model/1",
    python_model=DatabricksLLMWrapper(llm),
)


# COMMAND ----------

# Evaluate the model on some example questions
questions = pd.DataFrame(
    {
        "question": [
            "How do you use ARM devices ?"
        ]
    }
)

mlflow.evaluate(
    model=logged_model.model_uri,
    model_type="question-answering",
    data=questions,
)

mlflow.end_run()


# COMMAND ----------

# Register the model in the Model Registry
# Register the model under a name (e.g., "databricks_llm_model")
model_name = "databricks_llm_model"
model_uri = logged_model.model_uri

# Register the model in the registry
model_version = mlflow.register_model(model_uri, model_name)

# Print the registered model details
print(f"Model {model_name} version {model_version.version} registered successfully.")

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

class DatabricksLLMWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, llm):
        self.llm = llm  # Store the ChatDatabricks instance

    def predict(self, model_input):
        questions_list = model_input["question"].tolist()
        responses = self.llm.invoke(questions_list)  # Returns AIMessage objects
        
        # Extract the text content from AIMessage objects
        return responses
    
model = DatabricksLLMWrapper(llm)
questions = pd.DataFrame(
    {
        "question": [
            "How do you train an LLM ?",
            "What are the components of a mobile phone ?"
        ]
    }
)
res = model.predict(questions)

print(res)