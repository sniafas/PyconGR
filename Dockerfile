FROM public.ecr.aws/lambda/python:3.10


# pinpoint poetry version
ARG POETRY_VERSION=1.1.7

# install poetry and opencv libraries
RUN pip install "poetry==$POETRY_VERSION"
    
# Install Python dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# install dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    rm -rf "$(pip cache dir)" && \
    rm -rf /var/lib/apt/lists/*

# Copy function code
COPY src/lambda_function.py .
COPY src src

# Set the command to run the Lambda function handler
CMD ["lambda_function.handler"]

