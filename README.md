# AI Competitor Intelligence Agent Team

## Overview
The **AI Competitor Intelligence Agent Team** is a Streamlit-based application that helps businesses analyze their competitors by extracting structured data from competitor websites and generating insights using AI.

## Features
- **Competitor Discovery:** Identify competitors based on a given company URL or description.
- **Search Engine Options:** Choose between **Perplexity AI - Sonar Pro** and **Exa AI** to find competitor URLs.
- **Automated Data Extraction:** Use Firecrawl to scrape relevant competitor details.
- **AI-powered Analysis:** Extract insights such as pricing, key features, tech stack, marketing focus, and customer feedback.
- **Comparison Report:** Generate a structured competitor comparison table.

## Installation
### Prerequisites
Ensure you have Python installed on your system. Install required dependencies using:
```bash
pip install streamlit exa_py agno pandas requests firecrawl pydantic
```

### Running the App
```bash
streamlit run app.py
```

## Usage
1. **Enter API Keys:** Provide API keys for OpenAI, Firecrawl, and the selected search engine (Perplexity AI or Exa AI) in the sidebar.
2. **Enter Company Details:** Input a company URL and/or a brief description.
3. **Fetch Competitors:** The app will retrieve competitor URLs using the selected search engine.
4. **Analyze Data:** Extract detailed information about competitors.
5. **View Report:** Compare competitors using an AI-generated structured table.

## API Dependencies
- **OpenAI GPT-4o** (For AI-based analysis)
- **Firecrawl API** (For website crawling and data extraction)
- **Perplexity AI - Sonar Pro** / **Exa AI** (For competitor search)
- **DuckDuckGo Search Tools** (For additional competitor discovery)

## License
This project is licensed under the MIT License.

## Contact
For queries and contributions, reach out to the project maintainer.

