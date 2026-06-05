# Use an official, lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Copy only the requirements first to leverage Docker cache
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
# (We add numpy explicitly here just in case it wasn't in your original requirements)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt numpy

# Copy the app directory (containing main.py and the .pkl files) into the container
COPY ./app /code/app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]