# ============================================
# Customer Churn Prediction
# Production Docker Image
# ============================================

FROM python:3.13-slim


# ============================================
# Environment configuration
# ============================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ============================================
# Working directory
# ============================================

WORKDIR /app


# ============================================
# Install Python dependencies
# ============================================

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# ============================================
# Copy application
# ============================================

COPY . .


# ============================================
# Expose FastAPI port
# ============================================

EXPOSE 8000


# ============================================
# Start application
# ============================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]