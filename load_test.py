"""
Load test for the RAG query endpoint.

Usage:
    # Install locust
    pip install locust

    # Run with web UI (open http://localhost:8089)
    locust -f load_test.py

    # Run headless: 10 users, spawn 2/sec, run for 60s
    locust -f load_test.py --headless -u 10 -r 2 -t 60s

Set environment variables:
    TARGET_HOST  — base URL of the service (default: http://localhost:8000)
    API_KEY      — value for X-API-Key header
"""

import os
import random
from pathlib import Path

from locust import HttpUser, task, between

API_KEY = os.getenv("API_KEY", "...")

SAMPLE_QUERIES = [
    "Who conducted the inspection?",
    "When was the inspection conducted?",
    "Where was the inspection conducted?",
    "Was the inspection good or bad?",
    "How many violations were recorded?",
    "Find the contact details of the inspector",
    "What were the violations recorded?",
    "Explain the reason for inspection.",
]

SAMPLE_USER_ACCESS = [
    ["aoka"],
    ["aoka", "client"],
    ["client", "contractor"],
    ["aoka", "client", "contractor"]
]

SAMPLE_TEXT_CONTENT = [
    "Inspector John Smith conducted a routine safety inspection at the downtown facility on March 1, 2026. No violations were found. All fire exits were accessible and properly marked.",
    "The quarterly environmental audit revealed two minor violations: improper waste labeling in storage room B and an expired MSDS sheet for cleaning chemicals. Corrective actions were assigned.",
    "Site visit report for the new construction project at 123 Main Street. Foundation work is 80% complete. Concrete pouring scheduled for next week. Safety compliance is satisfactory.",
    "Employee training records review completed. 95% of staff have completed mandatory safety training. Remaining 5% are scheduled for next month's session.",
    "Equipment maintenance log shows all heavy machinery passed inspection. Crane #4 requires bearing replacement within 30 days. All other equipment is operational.",
]

DATA_DIR = Path(__file__).parent / "app" / "data"
SAMPLE_FILES = list(DATA_DIR.glob("*.pdf"))

class RAGUser(HttpUser):
    """Simulates a user making RAG queries."""

    host = os.getenv("TARGET_HOST", "http://localhost:8000")
    wait_time = between(1, 3)  # seconds between requests per user

    def on_start(self):
        self.headers = {"X-API-Key": "...", "Content-Type": "application/json"}

    @task(3)
    def query_default_top_k(self):
        """Standard RAG query with default top_k."""
        self.client.post(
            "/query/",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "user_access": random.choice(SAMPLE_USER_ACCESS),
            },
            headers=self.headers,
        )

    @task(1)
    def query_with_chat_history(self):
        """RAG query with chat history context."""
        self.client.post(
            "/query/",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "user_access": random.choice(SAMPLE_USER_ACCESS),
                "top_k": random.choice([3, 5, 10]),
                "chat_history": [
                    {"role": "user", "content": "What is the company policy?"},
                    {"role": "assistant", "content": "Could you be more specific about which policy?"},
                ],
            },
            headers=self.headers,
        )

    @task(1)
    def health_check(self):
        """Lightweight health check to measure baseline latency."""
        self.client.get("/health", headers=self.headers)

    @task(3)
    def search_only(self):
        """Directly test DB concurrency via search endpoint."""
        self.client.post(
            "/search/",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "user_access": random.choice(SAMPLE_USER_ACCESS),
                "top_k": 5,
            },
            headers=self.headers,
        )

    @task(2)
    def ingest_text(self):
        """Test text ingestion (LLM refinement + embedding + DB insert)."""
        self.client.post(
            "/ingest/text",
            json={
                "content": random.choice(SAMPLE_TEXT_CONTENT),
                "user_access": random.choice(SAMPLE_USER_ACCESS),
                "klass": "LoadTest",
                "record_id": str(random.randint(1, 10000)),
            },
            headers=self.headers,
        )

    @task(1)
    def ingest_file(self):
        """Test file ingestion with real PDFs from app/data/."""
        if not SAMPLE_FILES:
            return
        pdf = random.choice(SAMPLE_FILES)
        self.client.post(
            "/ingest/",
            files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")},
            data={
                "user_access": ",".join(random.choice(SAMPLE_USER_ACCESS)),
                "klass": "LoadTest",
                "record_id": str(random.randint(1, 10000)),
            },
            headers={"X-API-Key": API_KEY},
        )
