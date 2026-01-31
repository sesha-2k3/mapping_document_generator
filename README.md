# Informatica PowerCenter Mapping Document Generator

A Streamlit application that converts Informatica PowerCenter XML exports into human-readable mapping documentation with **LLM-powered plain English transformation logic** (AI generated, will be updated in the future).

## Features

- **Complete XML Parsing**: Parses all PowerCenter components (sources, targets, mappings, transformations, connectors)
- **Full Transformation Support**: Handles all 15 major transformation types
- **LLM-Powered Descriptions**: Uses Llama 3.3 70B to generate clear, business-friendly descriptions
- **Lineage Tracing**: Traces data flow from target columns back to source columns
- **CSV Export**: Generates mapping document with transformation logic
- **Exceptions Report**: QA report for unmapped columns and issues
- **ZIP Download**: Bundle all artifacts for easy sharing

## Quick Start

### 1. Get Groq API Key (Free)

1. Go to [console.groq.com](https://console.groq.com/)
2. Sign up for a free account
3. Create an API key

### 2. Setup Environment

```bash
# Clone or create project directory
mkdir informatica-mapper
cd informatica-mapper

# Create .env file with your API key
echo "GROQ_API_KEY=your_api_key_here" > .env

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Open in Browser

Navigate to `http://localhost:8501`

## File Structure

```
informatica-mapper/
├── app.py              # Main Streamlit application
├── config.py           # Configuration and system prompts
├── requirements.txt    # Python dependencies
├── .env               # API key (create this)
└── README.md          # This file
```

## Supported Transformations

| Transformation | Description |
|----------------|-------------|
| Source Qualifier | Source filtering, SQL overrides |
| Expression | Field calculations and derivations |
| Filter | Row-level filtering |
| Aggregator | COUNT, SUM, AVG, MIN, MAX, etc. |
| Lookup Procedure | Connected and unconnected lookups |
| Joiner | All join types (Inner, Outer) |
| Router | Conditional row routing |
| Union | Combining multiple sources |
| Rank | Top/Bottom N selection |
| Sorter | Sorting with optional distinct |
| Sequence Generator | Auto-incrementing IDs |
| Update Strategy | INSERT/UPDATE/DELETE flags |
| Normalizer | Repeating field expansion |
| Stored Procedure | Database procedure calls |
| Transaction Control | Commit/rollback logic |

## Output Format

### Mapping Document (CSV)

| Column | Description |
|--------|-------------|
| Mapping Name | Informatica mapping name |
| Source Table | Source table/file or "Lookup: TABLE" |
| Source Column | Source column name(s) |
| Source Datatype | Source data type |
| Source Length | Source field length |
| Target Table | Target table name |
| Target Column | Target column name |
| Target Datatype | Target data type |
| Target Length | Target field length |
| Transformation Logic | **Plain English description** |
| Rule Type | Type classification |
| Status | OK, UNMAPPED |

### Rule Types

| Type | Description |
|------|-------------|
| Direct | Pass-through mapping |
| Direct + Filter | Pass-through with filtering |
| Derived | Expression-based calculation |
| Lookup | Value from lookup table |
| Aggregated | Aggregation function result |
| Joined | Data from Joiner |
| Routed | Data from Router |
| Unioned | Combined sources |
| Ranked | Top/Bottom N rows |
| Sequence | Generated ID |
| System | System variable |

## Example Output

| Target Column | Transformation Logic | Rule Type |
|---------------|---------------------|-----------|
| SSN | Map SSN directly from source file. Include only records where SSN contains a valid numeric value. | Direct + Filter |
| PP_END_DATE | Convert PP_END_DATE from 'YYYYMMDD' string format to date type. Return null if the value is not a valid date. | Derived + Filter |
| PP_END_YEAR | Retrieve the pay period end year from the PAY_PERIOD reference table for the current pay period. | Lookup |
| RUN_DATE | Populate with the session start timestamp (the date and time when the ETL job began execution). | System |
| COUNTER_VALUE | Count the total number of detail records from the source file. | Aggregated |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| GROQ_API_KEY | Yes | Your Groq API key |

### Model Settings (config.py)

```python
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 4096
```

## Troubleshooting

### "GROQ_API_KEY not found"
- Create a `.env` file in the project directory
- Add: `GROQ_API_KEY=your_key_here`
- Restart the application

### XML Parse Error
- Ensure file is a valid PowerCenter XML export
- Try re-exporting from Designer
- Check file encoding (UTF-8 or Latin-1)

### Slow Generation
- Large files with many columns take longer
- Each column requires an LLM call
- Progress bar shows current status

### Rate Limits
- Groq free tier has rate limits
- If hitting limits, wait a moment and retry
- Consider upgrading for heavy usage

## API Usage

The application makes one LLM call per target column to generate descriptions. For a mapping with 50 target columns, expect ~50 API calls.

Groq's free tier typically allows:
- 30 requests per minute
- 14,400 requests per day

## License

MIT License - Feel free to modify and use as needed.

## Credits

- **LLM**: Llama 3.3 70B via [Groq](https://groq.com/)
- **UI**: [Streamlit](https://streamlit.io/)