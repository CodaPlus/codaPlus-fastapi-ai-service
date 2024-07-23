# Use a builder image to install dependencies and build wheels
FROM python:3.10-slim as builder

WORKDIR /usr/src/app

# Prevent Python from writing pyc files to disc and buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for your FastAPI app and build tools
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    curl gcc libpq-dev python3-dev musl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy only the files needed for installing Python dependencies
COPY pyproject.toml poetry.lock* ./

# Use Poetry to install dependencies to the system directly
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes && \
    pip wheel --no-cache-dir --wheel-dir /usr/src/app/wheels -r requirements.txt

# Start the final build stage
FROM python:3.10-slim

# Create a non-root user
RUN addgroup --system app && adduser --system --group app

# Set environment variables
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
WORKDIR $APP_HOME

EXPOSE 8000

# Install runtime dependencies (if any)
RUN apt-get update -qq && apt-get install -y --no-install-recommends libpq-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the Python dependencies from the builder stage
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*

# Copy the FastAPI project into the container
COPY . .

# Change ownership of the application files to the non-root user
RUN chown -R app:app $APP_HOME

# Switch to the non-root user
USER app