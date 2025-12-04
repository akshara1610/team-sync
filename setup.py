from setuptools import setup, find_packages

setup(
    name="teamsync",
    version="1.0.0",
    description="AI-Powered Meeting Agent for automated transcription, summarization, and task management",
    authors=[
        "Vrinda Ahuja <vva2113@columbia.edu>",
        "Sachi Kaushik <sk5476@columbia.edu>",
        "Akshara Pramod <ap4613@columbia.edu>"
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "sqlalchemy>=2.0.23",
        "chromadb>=0.4.22",
        "langchain>=0.1.0",
        "sentence-transformers>=2.2.2",
        "openai>=1.6.1",
        "jira>=3.5.2",
        "google-api-python-client>=2.108.0",
    ],
    entry_points={
        "console_scripts": [
            "teamsync=main:main",
        ],
    },
)
