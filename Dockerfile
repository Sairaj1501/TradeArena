# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 7860 (Required by Hugging Face Spaces)
EXPOSE 7860

# Run FastAPI app by default
# CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "inference.py"]