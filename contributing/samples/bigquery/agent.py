# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import os

from google.adk.agents import llm_agent
from google.adk.tools.bigquery import BigQueryCredentialsConfig
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig
from google.adk.tools.bigquery.config import WriteMode
import google.auth


class CredentialsType(Enum):
  """Write mode indicating what levels of write operations are allowed in BigQuery."""

  ADC = "application-default-credentials"
  """Application default credentials.

  See https://cloud.google.com/docs/authentication/provide-credentials-adc for more details.
  """

  SERVICE_ACCOUNT_KEY = "servcie-account-key"
  """Service Account Key credentials.

  See https://cloud.google.com/iam/docs/service-account-creds#key-types for more details.
  """

  OAUTH = "oauth"
  """OAuth credentials.

  See https://developers.google.com/identity/protocols/oauth2 for more details.
  """


# Define an appropriate credential type
CREDENTIALS_TYPE = CredentialsType.OAUTH


# Define BigQuery tool config
tool_config = BigQueryToolConfig(write_mode=WriteMode.ALLOWED)

if CREDENTIALS_TYPE == CredentialsType.ADC:
  # Initialize the tools to use the application default credentials.
  application_default_credentials, _ = google.auth.default()
  credentials_config = BigQueryCredentialsConfig(
      credentials=application_default_credentials
  )
elif CREDENTIALS_TYPE == CredentialsType.SERVICE_ACCOUNT_KEY:
  # Initialize the tools to use the credentials in the service account key.
  # If this flow is enabled, make sure to replace the file path with your own
  # service account key file
  creds, _ = google.auth.load_credentials_from_file("service_account_key.json")
  credentials_config = BigQueryCredentialsConfig(credentials=creds)
elif CREDENTIALS_TYPE == CredentialsType.OAUTH:
  # Initiaze the tools to do interactive OAuth
  # The environment variables OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET
  # must be set
  credentials_config = BigQueryCredentialsConfig(
      client_id=os.getenv("OAUTH_CLIENT_ID"),
      client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
  )
else:
  raise ValueError(
      f"Credential type {CREDENTIALS_TYPE} is not supported, please use one of"
      " the supported types."
  )

bigquery_toolset = BigQueryToolset(
    credentials_config=credentials_config, bigquery_tool_config=tool_config
)

# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = llm_agent.Agent(
    model="gemini-2.0-flash",
    name="hello_agent",
    description=(
        "Agent to answer questions about BigQuery data and models and execute"
        " SQL queries."
    ),
    instruction="""\
        You are a data science agent with access to several BigQuery tools.
        Make use of those tools to answer the user's questions.
    """,
    tools=[bigquery_toolset],
)
