"""
Configuration for Informatica Mapping Document Generator.
Contains constants, transformation configs, and system prompts.
"""

# -- Groq Model Configuration --
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# -- Supported Transformation Types --
TRANSFORMATION_TYPES = [
    "Source Qualifier",
    "Expression",
    "Filter",
    "Aggregator",
    "Lookup Procedure",
    "Joiner",
    "Router",
    "Union",
    "Rank",
    "Sorter",
    "Sequence Generator",
    "Update Strategy",
    "Normalizer",
    "Stored Procedure",
    "Transaction Control"
]

# -- XML Entity Replacements --
XML_ENTITY_MAP = {
    '&#xD;': '\n',
    '&#xA;': '\n',
    '&#x9;': '\t',
    '&lt;': '<',
    '&gt;': '>',
    '&amp;': '&',
    '&apos;': "'",
    '&quot;': '"',
    '&#x5c;': '\\'
}

# -- Rule Type Classifications --
RULE_TYPES = {
    'direct': 'Direct',
    'direct_filter': 'Direct + Filter',
    'derived': 'Derived',
    'derived_filter': 'Derived + Filter',
    'lookup': 'Lookup',
    'lookup_filter': 'Lookup + Filter',
    'aggregated': 'Aggregated',
    'aggregated_group': 'Aggregated + Group By',
    'joined': 'Joined',
    'routed': 'Routed',
    'unioned': 'Unioned',
    'ranked': 'Ranked',
    'sorted': 'Sorted',
    'normalized': 'Normalized',
    'sequence': 'Sequence',
    'update_strategy': 'Update Strategy',
    'system': 'System',
    'stored_procedure': 'Stored Procedure',
    'constant': 'Constant',
    'complex': 'Complex',
    'unmapped': 'Unmapped'
}

# -- Aggregation Functions --
AGGREGATION_FUNCTIONS = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'FIRST', 'LAST', 'MEDIAN', 'STDDEV', 'VARIANCE']

# -- System Variables --
SYSTEM_VARIABLES = [
    'SESSSTARTTIME', 'SESSENDTIME', 'SYSDATE', 'SYSTIMESTAMP',
    '$PMMappingName', '$PMSessionName', '$PMWorkflowName',
    '$PMRepositoryServiceName', '$PMIntegrationServiceName',
    '$PMFolderName', '$PMSessionRunMode'
]

# -- Expression Translation System Prompt --
EXPRESSION_TRANSLATION_PROMPT = '''You are an Informatica PowerCenter ETL expert. Your task is to translate Informatica transformation expressions into clear, plain English descriptions.

<task>
Convert Informatica PowerCenter expressions and transformation logic into human-readable business descriptions that non-technical users can understand.
</task>

<output_rules>
1. Return ONLY valid JSON, no markdown fences or explanations
2. Keep descriptions concise (1-2 sentences maximum)
3. Do NOT invent information - only describe what the expression actually does
4. Use business-friendly language, avoid technical jargon
5. If an expression is unclear, describe its apparent purpose
</output_rules>

<informatica_functions_reference>
STRING FUNCTIONS:
- LTRIM/RTRIM/TRIM: Remove leading/trailing spaces
- UPPER/LOWER: Convert case
- SUBSTR(str, start, length): Extract substring
- CONCAT or ||: Concatenate strings
- LPAD/RPAD: Pad string to length
- REPLACE/REPLACESTR: Replace characters
- INSTR: Find position of substring
- LENGTH: Get string length

DATE FUNCTIONS:
- TO_DATE(str, format): Convert string to date
- TO_CHAR(date, format): Convert date to string
- ADD_TO_DATE: Add interval to date
- DATE_DIFF: Difference between dates
- TRUNC(date): Truncate date to day
- GET_DATE_PART: Extract date component
- SYSTIMESTAMP/SYSDATE: Current date/time

NUMERIC FUNCTIONS:
- TO_DECIMAL/TO_INTEGER: Convert to number
- ROUND/TRUNC: Round or truncate
- ABS: Absolute value
- MOD: Modulo operation
- POWER/SQRT: Mathematical operations

CONDITIONAL FUNCTIONS:
- IIF(condition, true_value, false_value): If-then-else
- DECODE(value, match1, result1, ..., default): Value mapping
- NVL(value, default): Null handling
- ISNULL(value): Check if null
- IS_DATE/IS_NUMBER/IS_SPACES: Data validation

AGGREGATE FUNCTIONS:
- COUNT/SUM/AVG/MIN/MAX: Standard aggregations
- FIRST/LAST: First or last value in group
- MEDIAN/STDDEV/VARIANCE: Statistical functions

LOOKUP FUNCTIONS:
- :LKP.lookup_name(condition): Unconnected lookup call

SPECIAL:
- SETVARIABLE($$var, value): Set mapping variable
- ABORT(message): Abort session with error
- ERROR(message): Log error and continue
- MD5(value): Generate hash
</informatica_functions_reference>

<transformation_context>
When translating, consider the transformation type:
- Expression: Field-level calculations and derivations
- Filter: Row filtering conditions (TRUE = keep row)
- Aggregator: Grouping and aggregate calculations
- Lookup: Retrieving reference data
- Joiner: Combining data from multiple sources
- Router: Splitting data into multiple paths
- Rank: Selecting top/bottom N rows
- Update Strategy: Determining INSERT/UPDATE/DELETE operations
</transformation_context>

<examples>
INPUT: IIF(BALANCE > 10000, 'VIP', 'STANDARD')
OUTPUT: Assign 'VIP' status if balance exceeds 10,000, otherwise assign 'STANDARD'

INPUT: UPPER(TRIM(COUNTRY))
OUTPUT: Convert country to uppercase after removing leading and trailing spaces

INPUT: IIF(IS_DATE(PP_END_DATE, 'YYYYMMDD'), TO_DATE(PP_END_DATE, 'YYYYMMDD'))
OUTPUT: Convert PP_END_DATE from 'YYYYMMDD' string format to date type; return null if invalid

INPUT: TO_DECIMAL(TO_CHAR(lkp_PP_END_YEAR) || LPAD(TO_CHAR(lkp_PP_NUM), 2, '0'))
OUTPUT: Combine pay period year and zero-padded period number into a single numeric identifier (e.g., 201905)

INPUT: DECODE(TRUE, IS_NUMBER(SSN), 'D', 'NO')
OUTPUT: Mark record as 'D' (detail) if SSN contains only numbers, otherwise mark as 'NO'

INPUT: SETVARIABLE($$MAP_SUBJECT, v_SUBJECT)
OUTPUT: Store the computed subject line in a mapping variable for use in downstream processing

INPUT: COUNT(SSN)
OUTPUT: Count the total number of SSN values

INPUT: SUM(AMOUNT) with GROUP BY CUSTOMER_ID
OUTPUT: Calculate the total amount for each customer

INPUT: SESSSTARTTIME
OUTPUT: Use the session start timestamp (when the ETL job began)

INPUT: $PMMappingName
OUTPUT: Use the current mapping name as the value

INPUT: RECORD_TYPE_FLAG = 'D'
OUTPUT: Include only detail records (where record type flag equals 'D')

INPUT: CURR_PP_FLAG = in_CURR_PP_FLAG
OUTPUT: Match records where the current pay period flag equals the input value

INPUT: :LKP.LKP_CUSTOMER(CUST_ID)
OUTPUT: Look up customer information using customer ID from the LKP_CUSTOMER lookup
</examples>

<response_format>
Respond with valid JSON only:
{
    "translations": [
        {"index": 1, "plain_english": "description here"},
        {"index": 2, "plain_english": "description here"}
    ]
}
</response_format>
'''

# -- Lineage Summary System Prompt --
LINEAGE_SUMMARY_PROMPT = '''You are an Informatica PowerCenter ETL expert. Your task is to create a concise plain English summary of a data transformation lineage.

<task>
Given the lineage information for a target column (source columns, transformation path, expressions, filters, lookups, etc.), create a single cohesive description that explains how the target column gets its value.
</task>

<output_rules>
1. Return ONLY valid JSON with a single "description" field
2. Keep the description to 1-3 sentences maximum
3. Start with the most important information (source or derivation)
4. Mention filters only if they significantly affect the data
5. Use plain business language
6. Do NOT invent information
</output_rules>

<examples>
INPUT LINEAGE:
- Source: U0287D01.SSN
- Filter: VALID_RECORD_FLAG = TRUE (where flag = IS_NUMBER(SSN))
- Path: SQ → exp_Initial → fil_Valid_Records → exp_Convert → exp_Final → Target

OUTPUT:
{"description": "Map SSN directly from source file. Include only records where SSN contains a valid numeric value."}

---

INPUT LINEAGE:
- Source: Lookup from PAY_PERIOD table
- Lookup condition: CURR_PP_FLAG = 'Y'
- Expression: lkp_PP_END_YEAR (pass-through)

OUTPUT:
{"description": "Retrieve the pay period end year from the PAY_PERIOD reference table for the current pay period (where CURR_PP_FLAG = 'Y')."}

---

INPUT LINEAGE:
- Source: U0287D01.PP_END_DATE (string, 8 chars)
- Expression: IIF(IS_DATE(PP_END_DATE, 'YYYYMMDD'), TO_DATE(PP_END_DATE, 'YYYYMMDD'))
- Filter: VALID_RECORD_FLAG = TRUE

OUTPUT:
{"description": "Convert PP_END_DATE from 'YYYYMMDD' string format to date type. Return null if the value is not a valid date. Include only records with valid SSN."}

---

INPUT LINEAGE:
- No source column
- Expression: SESSSTARTTIME

OUTPUT:
{"description": "Populate with the session start timestamp (the date and time when the ETL job began execution)."}

---

INPUT LINEAGE:
- Source: Multiple from U0287D01 (SSN used for counting)
- Aggregation: COUNT(SSN), Scope: All Input
- Filter: RECORD_TYPE_FLAG = 'D'

OUTPUT:
{"description": "Count the total number of detail records from the source file. Only records with numeric SSN values are counted."}
</examples>

<response_format>
{"description": "your concise description here"}
</response_format>
'''

# -- Batch Translation Prompt Builder --
def build_expression_batch_prompt(expressions: list[dict]) -> str:
    """Build the user prompt for batch expression translation."""
    expr_lines = []
    for i, expr_data in enumerate(expressions, 1):
        field = expr_data.get('field', 'unknown')
        expr = expr_data.get('expression', '')[:300]  # Limit length
        trans_type = expr_data.get('transformation_type', '')
        context = f" [{trans_type}]" if trans_type else ""
        expr_lines.append(f"{i}. Field: {field}{context}\n   Expression: {expr}")
    
    return f"""<request>
Translate the following Informatica PowerCenter expressions to plain English.
</request>

<expressions>
{chr(10).join(expr_lines)}
</expressions>

<instructions>
1. Analyze each expression carefully
2. Provide a clear, concise business description
3. Return valid JSON with translations array
</instructions>

Translate now:"""


def build_lineage_summary_prompt(lineage: dict) -> str:
    """Build the user prompt for lineage summary generation."""
    parts = []
    
    # Source columns
    if lineage.get('source_columns'):
        sources = [f"{s['table']}.{s['column']}" for s in lineage['source_columns']]
        parts.append(f"Source columns: {', '.join(sources)}")
    
    # Lookups
    if lineage.get('lookups'):
        for lkp in lineage['lookups']:
            parts.append(f"Lookup: {lkp['table']} (condition: {lkp.get('condition', 'N/A')})")
    
    # Aggregations
    if lineage.get('aggregations'):
        for agg in lineage['aggregations']:
            scope = agg.get('scope', 'All Input')
            parts.append(f"Aggregation: {agg['expression']} (scope: {scope})")
    
    # Filters
    if lineage.get('filters'):
        for f in lineage['filters']:
            parts.append(f"Filter: {f['condition']}")
    
    # Key expressions
    if lineage.get('expressions'):
        for expr in lineage['expressions'][:3]:  # Limit to 3
            if expr.get('expression') and expr['expression'] != expr.get('field', ''):
                parts.append(f"Expression: {expr['expression'][:150]}")
    
    # Joiners
    if lineage.get('joiners'):
        for j in lineage['joiners']:
            parts.append(f"Join: {j.get('join_type', 'Normal')} join")
    
    # Transformation path
    if lineage.get('transformation_path'):
        path = ' → '.join([t['name'] for t in lineage['transformation_path']])
        parts.append(f"Path: {path}")
    
    lineage_text = '\n'.join(f"- {p}" for p in parts) if parts else "- No lineage information available"
    
    return f"""<request>
Create a plain English description for this target column's data lineage.
</request>

<target>
Target Column: {lineage.get('target_field', 'unknown')}
</target>

<lineage>
{lineage_text}
</lineage>

<instructions>
Summarize how this target column gets its value in 1-3 sentences.
Focus on: source of data, key transformations, any filtering applied.
</instructions>

Generate the description now:"""