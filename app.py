"""
Informatica PowerCenter XML to Mapping Document Converter
Generates mapping documentation with plain English transformation logic using LLM.
"""

import os
import io
import re
import json
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict

import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from config import (
    MODEL_NAME, TEMPERATURE, MAX_TOKENS,
    XML_ENTITY_MAP, RULE_TYPES, SYSTEM_VARIABLES,
    AGGREGATION_FUNCTIONS, TRANSFORMATION_TYPES,
    EXPRESSION_TRANSLATION_PROMPT, LINEAGE_SUMMARY_PROMPT,
    build_expression_batch_prompt, build_lineage_summary_prompt
)

load_dotenv()

st.set_page_config(page_title="Informatica Mapping Document Generator", layout="wide")


# -- XML Parsing Functions --

def decode_xml_entities(text: str) -> str:
    """Decode XML entities and escape sequences."""
    if not text:
        return ""
    for entity, char in XML_ENTITY_MAP.items():
        text = text.replace(entity, char)
    return text.strip()


def parse_transformation(trans: ET.Element) -> dict:
    """Parse a single transformation element with all attributes."""
    fields = []
    for field in trans.findall('TRANSFORMFIELD'):
        field_data = {
            'name': field.get('NAME'),
            'datatype': field.get('DATATYPE'),
            'porttype': field.get('PORTTYPE', ''),
            'expression': decode_xml_entities(field.get('EXPRESSION', '')),
            'precision': field.get('PRECISION', ''),
            'scale': field.get('SCALE', ''),
            'default_value': field.get('DEFAULTVALUE', ''),
        }
        # Type-specific attributes
        if trans.get('TYPE') == 'Joiner':
            field_data['master'] = field.get('MASTER', 'NO')
        if trans.get('TYPE') == 'Rank':
            field_data['rankport'] = field.get('RANKPORT', 'NO')
        fields.append(field_data)
    
    attributes = {}
    for attr in trans.findall('TABLEATTRIBUTE'):
        attr_name = attr.get('NAME')
        attr_value = decode_xml_entities(attr.get('VALUE', ''))
        if attr_value:
            attributes[attr_name] = attr_value
    
    groups = []
    for group in trans.findall('GROUP'):
        groups.append({
            'name': group.get('NAME', ''),
            'expression': decode_xml_entities(group.get('EXPRESSION', '')),
            'order': group.get('ORDER', '0')
        })
    
    return {
        'name': trans.get('NAME'),
        'type': trans.get('TYPE'),
        'description': trans.get('DESCRIPTION', ''),
        'reusable': trans.get('REUSABLE', 'NO'),
        'fields': fields,
        'attributes': attributes,
        'groups': groups
    }


def parse_xml(xml_content: str) -> tuple[dict | None, str | None]:
    """Parse Informatica PowerCenter XML and extract all components."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        return None, f"XML Parse Error: {e}"
    
    result = {
        'sources': {},
        'targets': {},
        'mappings': {},
        'reusable_transformations': {}
    }
    
    for folder in root.iter('FOLDER'):
        folder_name = folder.get('NAME', 'Unknown')
        
        # Parse SOURCES
        for source in folder.findall('SOURCE'):
            src_name = source.get('NAME')
            fields = [{
                'name': f.get('NAME'),
                'datatype': f.get('DATATYPE'),
                'precision': f.get('PRECISION', ''),
                'scale': f.get('SCALE', ''),
                'length': f.get('LENGTH', f.get('PRECISION', ''))
            } for f in source.findall('SOURCEFIELD')]
            
            result['sources'][src_name] = {
                'name': src_name,
                'type': source.get('DATABASETYPE', 'Unknown'),
                'folder': folder_name,
                'fields': fields
            }
        
        # Parse TARGETS
        for target in folder.findall('TARGET'):
            tgt_name = target.get('NAME')
            fields = [{
                'name': f.get('NAME'),
                'datatype': f.get('DATATYPE'),
                'precision': f.get('PRECISION', ''),
                'scale': f.get('SCALE', ''),
                'length': f.get('PRECISION', '')
            } for f in target.findall('TARGETFIELD')]
            
            result['targets'][tgt_name] = {
                'name': tgt_name,
                'type': target.get('DATABASETYPE', 'Unknown'),
                'folder': folder_name,
                'fields': fields
            }
        
        # Parse REUSABLE TRANSFORMATIONS
        for trans in folder.findall('TRANSFORMATION'):
            if trans.get('REUSABLE', 'NO') == 'YES':
                t_name = trans.get('NAME')
                result['reusable_transformations'][t_name] = parse_transformation(trans)
        
        # Parse MAPPINGS
        for mapping in folder.findall('MAPPING'):
            map_name = mapping.get('NAME')
            
            transformations = {}
            for trans in mapping.findall('TRANSFORMATION'):
                t_name = trans.get('NAME')
                transformations[t_name] = parse_transformation(trans)
            
            instances = {}
            for inst in mapping.findall('INSTANCE'):
                assoc_elem = inst.find('ASSOCIATED_SOURCE_INSTANCE')
                instances[inst.get('NAME')] = {
                    'name': inst.get('NAME'),
                    'type': inst.get('TYPE'),
                    'transformation_name': inst.get('TRANSFORMATION_NAME', inst.get('NAME')),
                    'transformation_type': inst.get('TRANSFORMATION_TYPE', ''),
                    'reusable': inst.get('REUSABLE', 'NO'),
                    'associated_source': assoc_elem.get('NAME') if assoc_elem is not None else None
                }
            
            connectors = [{
                'from_instance': c.get('FROMINSTANCE'),
                'from_field': c.get('FROMFIELD'),
                'to_instance': c.get('TOINSTANCE'),
                'to_field': c.get('TOFIELD'),
                'from_type': c.get('FROMINSTANCETYPE', ''),
                'to_type': c.get('TOINSTANCETYPE', '')
            } for c in mapping.findall('CONNECTOR')]
            
            result['mappings'][map_name] = {
                'name': map_name,
                'description': mapping.get('DESCRIPTION', ''),
                'folder': folder_name,
                'transformations': transformations,
                'instances': instances,
                'connectors': connectors
            }
    
    return result, None


# -- Building the graph and then tracing the lineage --

def build_lineage_graph(mapping_data: dict) -> tuple[dict, dict]:
    """Build forward and reverse adjacency graphs from connectors."""
    forward = defaultdict(list)
    reverse = defaultdict(list)
    
    for conn in mapping_data['connectors']:
        from_node = (conn['from_instance'], conn['from_field'])
        to_node = (conn['to_instance'], conn['to_field'])
        forward[from_node].append(to_node)
        reverse[to_node].append(from_node)
    
    return dict(forward), dict(reverse)


def get_transformation_data(inst_name: str, inst_info: dict, mapping_data: dict, parsed_data: dict) -> dict:
    """Get transformation data, handling reusable transformations."""
    trans_name = inst_info.get('transformation_name', inst_name)
    
    if trans_name in mapping_data['transformations']:
        return mapping_data['transformations'][trans_name]
    if trans_name in parsed_data.get('reusable_transformations', {}):
        return parsed_data['reusable_transformations'][trans_name]
    return {}


def extract_transformation_info(lineage: dict, trans_data: dict, trans_type: str, trans_name: str, field: str) -> None:
    """Extract relevant information from a transformation based on its type."""
    attrs = trans_data.get('attributes', {})
    fields = trans_data.get('fields', [])
    
    if trans_type == 'Source Qualifier':
        if attrs.get('Source Filter'):
            lineage['filters'].append({'transformation': trans_name, 'condition': attrs['Source Filter'], 'type': 'Source Filter'})
        if attrs.get('Sql Query'):
            lineage['expressions'].append({'transformation': trans_name, 'field': field, 'expression': f"SQL: {attrs['Sql Query']}", 'porttype': 'SQL'})
    
    elif trans_type == 'Expression':
        for f in fields:
            if f['name'] == field and f.get('expression') and f.get('porttype') in ['OUTPUT', 'LOCAL VARIABLE', 'INPUT/OUTPUT']:
                lineage['expressions'].append({'transformation': trans_name, 'field': field, 'expression': f['expression'], 'porttype': f['porttype']})
    
    elif trans_type == 'Filter':
        cond = attrs.get('Filter Condition', '')
        if cond and cond not in [f['condition'] for f in lineage['filters']]:
            lineage['filters'].append({'transformation': trans_name, 'condition': cond, 'type': 'Filter'})
    
    elif trans_type == 'Aggregator':
        for f in fields:
            if f['name'] == field and f.get('expression'):
                lineage['aggregations'].append({
                    'transformation': trans_name, 'field': field, 'expression': f['expression'],
                    'porttype': f.get('porttype', ''), 'scope': attrs.get('Transformation Scope', 'All Input')
                })
    
    elif trans_type == 'Lookup Procedure':
        if attrs.get('Lookup table name'):
            lineage['lookups'].append({
                'transformation': trans_name, 'table': attrs['Lookup table name'],
                'condition': attrs.get('Lookup condition', ''), 'sql_override': attrs.get('Lookup Sql Override', ''), 'field': field
            })
    
    elif trans_type == 'Joiner':
        if trans_name not in [j['transformation'] for j in lineage['joiners']]:
            lineage['joiners'].append({
                'transformation': trans_name, 'join_type': attrs.get('Join Type', 'Normal'),
                'condition': attrs.get('Join Condition', ''),
                'master_fields': [f['name'] for f in fields if f.get('master') == 'YES']
            })
    
    elif trans_type == 'Router':
        if trans_name not in [r['transformation'] for r in lineage['routers']]:
            lineage['routers'].append({
                'transformation': trans_name,
                'groups': [{'name': g['name'], 'expression': g['expression']} for g in trans_data.get('groups', [])]
            })
    
    elif trans_type == 'Union':
        if trans_name not in [u['transformation'] for u in lineage['unions']]:
            lineage['unions'].append({'transformation': trans_name})
    
    elif trans_type == 'Rank':
        if trans_name not in [r['transformation'] for r in lineage['ranks']]:
            rank_port = next((f['name'] for f in fields if f.get('rankport') == 'YES'), None)
            lineage['ranks'].append({
                'transformation': trans_name, 'rank_port': rank_port,
                'top_bottom': attrs.get('Top/Bottom', 'Top'), 'number_of_ranks': attrs.get('Number Of Ranks', '1')
            })
    
    elif trans_type == 'Sorter':
        if trans_name not in [s['transformation'] for s in lineage['sorters']]:
            lineage['sorters'].append({'transformation': trans_name, 'distinct': attrs.get('Distinct', 'NO')})
    
    elif trans_type == 'Sequence Generator':
        if trans_name not in [s['transformation'] for s in lineage['sequences']]:
            lineage['sequences'].append({
                'transformation': trans_name, 'start_value': attrs.get('Start Value', '1'), 'increment': attrs.get('Increment By', '1')
            })
    
    elif trans_type == 'Update Strategy':
        if trans_name not in [u['transformation'] for u in lineage['update_strategies']]:
            update_expr = next((f['expression'] for f in fields if f.get('expression') and 'DD_' in f.get('expression', '')), '')
            lineage['update_strategies'].append({'transformation': trans_name, 'expression': update_expr})
    
    elif trans_type == 'Normalizer':
        if trans_name not in [n['transformation'] for n in lineage['normalizers']]:
            lineage['normalizers'].append({'transformation': trans_name})
    
    elif trans_type == 'Stored Procedure':
        if trans_name not in [s['transformation'] for s in lineage['stored_procedures']]:
            lineage['stored_procedures'].append({'transformation': trans_name, 'procedure_name': attrs.get('Stored Procedure Name', '')})


def trace_lineage(target_instance: str, target_field: str, reverse_graph: dict, mapping_data: dict, parsed_data: dict) -> dict:
    """Trace lineage from target back to source(s)."""
    lineage = {
        'target_instance': target_instance, 'target_field': target_field,
        'source_columns': [], 'transformation_path': [], 'expressions': [],
        'filters': [], 'lookups': [], 'aggregations': [], 'joiners': [],
        'routers': [], 'unions': [], 'ranks': [], 'sorters': [],
        'sequences': [], 'update_strategies': [], 'normalizers': [],
        'stored_procedures': [], 'status': 'OK'
    }
    
    visited = set()
    stack = [(target_instance, target_field, [])]
    instances = mapping_data['instances']
    
    while stack:
        inst, field, path = stack.pop()
        if (inst, field) in visited:
            continue
        visited.add((inst, field))
        
        inst_info = instances.get(inst, {})
        inst_type = inst_info.get('type', '')
        trans_name = inst_info.get('transformation_name', inst)
        trans_type = inst_info.get('transformation_type', '')
        
        # Check if source
        if inst_type == 'SOURCE' or trans_type == 'Source Definition':
            if trans_name in parsed_data['sources']:
                src_info = parsed_data['sources'][trans_name]
                lineage['source_columns'].append({
                    'table': trans_name, 'column': field,
                    'datatype': next((f['datatype'] for f in src_info['fields'] if f['name'] == field), ''),
                    'length': next((f['length'] for f in src_info['fields'] if f['name'] == field), '')
                })
            continue
        
        trans_data = get_transformation_data(inst, inst_info, mapping_data, parsed_data)
        actual_type = trans_data.get('type', trans_type)
        
        if trans_name not in [p['name'] for p in lineage['transformation_path']]:
            lineage['transformation_path'].append({'name': trans_name, 'type': actual_type})
        
        extract_transformation_info(lineage, trans_data, actual_type, trans_name, field)
        
        for up_inst, up_field in reverse_graph.get((inst, field), []):
            stack.append((up_inst, up_field, path + [trans_name]))
    
    # Set status
    if not lineage['source_columns'] and not lineage['lookups'] and not lineage['sequences']:
        has_system = any(any(sv in e.get('expression', '') for sv in SYSTEM_VARIABLES) for e in lineage['expressions'])
        if not has_system:
            lineage['status'] = 'UNMAPPED'
    
    return lineage


# -- LLM Functions --

def call_llm(system_prompt: str, user_prompt: str) -> tuple[str | None, str | None]:
    """Call Groq API for content generation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None, "GROQ_API_KEY not found in .env file"
    
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return None, f"API Error: {e}"


def parse_llm_json(content: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences."""
    content = content.strip()
    for prefix in ('```json', '```'):
        if content.startswith(prefix):
            content = content[len(prefix):]
    if content.endswith('```'):
        content = content[:-3]
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def translate_expressions_batch(expressions: list[dict]) -> dict[int, str]:
    """Translate a batch of expressions using LLM."""
    if not expressions:
        return {}
    
    user_prompt = build_expression_batch_prompt(expressions)
    content, error = call_llm(EXPRESSION_TRANSLATION_PROMPT, user_prompt)
    
    if error or not content:
        return {}
    
    result = parse_llm_json(content)
    if result and 'translations' in result:
        return {t['index']: t['plain_english'] for t in result['translations']}
    return {}


def generate_lineage_description(lineage: dict) -> str | None:
    """Generate a plain English description of a column's lineage using LLM."""
    user_prompt = build_lineage_summary_prompt(lineage)
    content, error = call_llm(LINEAGE_SUMMARY_PROMPT, user_prompt)
    
    if error or not content:
        return None
    
    result = parse_llm_json(content)
    if result and 'description' in result:
        return result['description']
    return None


# -- Document Generation --

def is_aggregation_expression(expression: str) -> bool:
    """Check if an expression contains an aggregation function."""
    if not expression:
        return False
    exp_upper = expression.upper()
    return any(f"{func}(" in exp_upper for func in AGGREGATION_FUNCTIONS)


def determine_rule_type(lineage: dict) -> str:
    """Determine the rule type based on lineage information."""
    has_filter = len(lineage.get('filters', [])) > 0
    
    # Sequence Generator
    if lineage.get('sequences'):
        return RULE_TYPES['sequence']
    
    # Stored Procedure
    if lineage.get('stored_procedures'):
        return RULE_TYPES['stored_procedure']
    
    # Joiner
    if lineage.get('joiners'):
        return RULE_TYPES['joined'] + (' + Filter' if has_filter else '')
    
    # Router
    if lineage.get('routers'):
        return RULE_TYPES['routed']
    
    # Union
    if lineage.get('unions'):
        return RULE_TYPES['unioned']
    
    # Rank
    if lineage.get('ranks'):
        return RULE_TYPES['ranked']
    
    # Sorter
    if lineage.get('sorters'):
        return RULE_TYPES['sorted']
    
    # Normalizer
    if lineage.get('normalizers'):
        return RULE_TYPES['normalized']
    
    # Update Strategy
    if lineage.get('update_strategies'):
        return RULE_TYPES['update_strategy']
    
    # Aggregator - also check expressions for aggregation functions
    if lineage.get('aggregations'):
        return RULE_TYPES['aggregated']
    
    # Check expressions for aggregation functions not caught by aggregations list
    for expr in lineage.get('expressions', []):
        if is_aggregation_expression(expr.get('expression', '')):
            return RULE_TYPES['aggregated']
    
    # Lookup
    if lineage.get('lookups'):
        return RULE_TYPES['lookup_filter'] if has_filter else RULE_TYPES['lookup']
    
    # System variables
    for expr in lineage.get('expressions', []):
        exp_text = expr.get('expression', '')
        if any(sv in exp_text for sv in SYSTEM_VARIABLES):
            return RULE_TYPES['system']
    
    # Derived expressions
    has_derivation = False
    for expr in lineage.get('expressions', []):
        exp_text = expr.get('expression', '')
        field = expr.get('field', '')
        if exp_text and exp_text.strip().upper() != field.upper():
            has_derivation = True
            break
    
    if has_derivation:
        return RULE_TYPES['derived_filter'] if has_filter else RULE_TYPES['derived']
    
    # Filter only
    if has_filter:
        return RULE_TYPES['direct_filter']
    
    # Direct mapping
    if lineage.get('source_columns'):
        return RULE_TYPES['direct']
    
    # Unmapped
    if lineage.get('status') == 'UNMAPPED':
        return RULE_TYPES['unmapped']
    
    return RULE_TYPES['direct']


def generate_mapping_document(parsed_data: dict, progress_callback=None) -> tuple[pd.DataFrame, list[dict]]:
    """Generate the mapping document with LLM-powered descriptions."""
    rows = []
    exceptions = []
    all_lineages = []
    
    total_mappings = len(parsed_data['mappings'])
    current = 0
    
    # First pass: collect all lineages
    for map_name, mapping in parsed_data['mappings'].items():
        if progress_callback:
            progress_callback(current / (total_mappings * 2), f"Tracing lineage: {map_name}...")
        
        _, reverse_graph = build_lineage_graph(mapping)
        
        target_instances = [(n, d) for n, d in mapping['instances'].items() if d['type'] == 'TARGET']
        
        for target_inst, target_inst_data in target_instances:
            target_name = target_inst_data['transformation_name']
            target_def = parsed_data['targets'].get(target_name, {})
            
            for target_field in target_def.get('fields', []):
                field_name = target_field['name']
                lineage = trace_lineage(target_inst, field_name, reverse_graph, mapping, parsed_data)
                lineage['mapping_name'] = map_name
                lineage['target_table'] = target_name
                lineage['target_field_info'] = target_field
                all_lineages.append(lineage)
        
        current += 1
    
    # Second pass: generate descriptions with LLM
    total_lineages = len(all_lineages)
    for i, lineage in enumerate(all_lineages):
        if progress_callback:
            progress_callback((total_mappings + (i / total_lineages) * total_mappings) / (total_mappings * 2),
                            f"Generating descriptions: {lineage['mapping_name']}...")
        
        # Generate description using LLM
        description = generate_lineage_description(lineage)
        
        if not description:
            description = "Unable to generate description"
            if lineage['status'] == 'UNMAPPED':
                description = "No upstream mapping found"
                exceptions.append({
                    'type': 'UNMAPPED', 'mapping': lineage['mapping_name'],
                    'target_table': lineage['target_table'], 'target_column': lineage['target_field'],
                    'message': 'Target column has no upstream connector'
                })
        
        # Determine source info
        if lineage['source_columns']:
            src = lineage['source_columns'][0]
            src_table, src_column = src['table'], src['column']
            src_datatype, src_length = src['datatype'], src['length']
            if len(lineage['source_columns']) > 1:
                src_column = ', '.join(set(s['column'] for s in lineage['source_columns']))
        elif lineage['lookups']:
            lkp = lineage['lookups'][0]
            src_table, src_column = f"Lookup: {lkp['table']}", lkp['field']
            src_datatype, src_length = '', ''
        elif lineage['sequences']:
            src_table, src_column = '(sequence)', lineage['sequences'][0]['transformation']
            src_datatype, src_length = 'number', ''
        else:
            src_table, src_column, src_datatype, src_length = '(derived/system)', '', '', ''
        
        target_field = lineage['target_field_info']
        rows.append({
            'Mapping Name': lineage['mapping_name'],
            'Source Table': src_table,
            'Source Column': src_column,
            'Source Datatype': src_datatype,
            'Source Length': src_length,
            'Target Table': lineage['target_table'],
            'Target Column': lineage['target_field'],
            'Target Datatype': target_field['datatype'],
            'Target Length': target_field.get('precision', ''),
            'Transformation Logic': description,
            'Rule Type': determine_rule_type(lineage),
            'Status': lineage['status']
        })
    
    return pd.DataFrame(rows), exceptions


def generate_exceptions_report(exceptions: list[dict], parsed_data: dict) -> str:
    """Generate the exceptions/QA report."""
    lines = [
        "=" * 70,
        "INFORMATICA MAPPING DOCUMENT - EXCEPTIONS REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70, "",
        "SUMMARY", "-" * 40,
        f"Total Mappings: {len(parsed_data['mappings'])}",
        f"Total Sources: {len(parsed_data['sources'])}",
        f"Total Targets: {len(parsed_data['targets'])}",
        f"Total Exceptions: {len(exceptions)}", "",
        "MAPPINGS", "-" * 40
    ]
    
    for name, mapping in parsed_data['mappings'].items():
        lines.append(f"\n  {name}")
        lines.append(f"    Transformations: {len(mapping.get('transformations', {}))}")
        type_counts = defaultdict(int)
        for t in mapping.get('transformations', {}).values():
            type_counts[t.get('type', 'Unknown')] += 1
        if type_counts:
            lines.append(f"    Types: {', '.join(f'{t}:{c}' for t, c in sorted(type_counts.items()))}")
    
    lines.extend(["", "SOURCES", "-" * 40])
    for name, src in parsed_data['sources'].items():
        lines.append(f"  {name} ({src.get('type', 'Unknown')}) - {len(src.get('fields', []))} fields")
    
    lines.extend(["", "TARGETS", "-" * 40])
    for name, tgt in parsed_data['targets'].items():
        lines.append(f"  {name} ({tgt.get('type', 'Unknown')}) - {len(tgt.get('fields', []))} fields")
    
    if exceptions:
        lines.extend(["", "EXCEPTIONS", "-" * 40])
        by_type = defaultdict(list)
        for exc in exceptions:
            by_type[exc['type']].append(exc)
        for exc_type, exc_list in by_type.items():
            lines.append(f"\n[{exc_type}] - {len(exc_list)} issue(s)")
            for exc in exc_list:
                lines.append(f"  {exc['mapping']}: {exc['target_table']}.{exc['target_column']}")
    else:
        lines.append("\nNo exceptions found.")
    
    lines.extend(["", "=" * 70, "END OF REPORT", "=" * 70])
    return "\n".join(lines)


def create_download_bundle(df: pd.DataFrame, report: str, base_name: str) -> bytes:
    """Create a ZIP file containing all artifacts."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_name}_mapping_document.csv", df.to_csv(index=False))
        zf.writestr(f"{base_name}_exceptions_report.txt", report)
    buffer.seek(0)
    return buffer.getvalue()


# -- UI Components --

def render_sidebar() -> None:
    """Render sidebar with information."""
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool parses **Informatica PowerCenter XML** exports and generates:
        - CSV mapping document with plain English transformation logic
        - Exceptions report for QA
        """)
        
        st.divider()
        st.subheader("Supported Transformations")
        for t in TRANSFORMATION_TYPES:
            st.markdown(f"- {t}")
        
        st.divider()
        st.subheader("How to Use")
        st.markdown("""
        1. **Export** mapping from PowerCenter Designer as XML
        2. **Upload** the XML file
        3. **Review** the detected mappings
        4. **Generate** the documentation
        5. **Download** CSV and report
        """)
        
        st.divider()
        st.caption("Powered by Llama 3.3 70B via Groq")


def render_mapping_expanders(parsed_data: dict) -> None:
    """Render expandable details for each mapping."""
    for map_name, mapping in parsed_data['mappings'].items():
        with st.expander(f"{map_name}"):
            desc = mapping.get('description', 'No description')
            if desc:
                st.write(f"**Description:** {desc}")
            
            type_counts = defaultdict(int)
            for t in mapping.get('transformations', {}).values():
                type_counts[t.get('type', 'Unknown')] += 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Transformations:** {len(mapping.get('transformations', {}))}")
            with col2:
                st.write(f"**Connectors:** {len(mapping.get('connectors', []))}")
            
            if type_counts:
                st.write("**Transformation Types:**")
                for t_type, count in sorted(type_counts.items()):
                    st.write(f"  - {t_type}: {count}")


def render_results(df: pd.DataFrame, report: str, exceptions: list[dict]) -> None:
    """Render the results section."""
    # Filters
    st.subheader("Mapping Document")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_mapping = st.selectbox("Filter by Mapping", ["All"] + list(df['Mapping Name'].unique()))
    with col2:
        selected_rule = st.selectbox("Filter by Rule Type", ["All"] + list(df['Rule Type'].unique()))
    with col3:
        selected_status = st.selectbox("Filter by Status", ["All"] + list(df['Status'].unique()))
    
    filtered = df.copy()
    if selected_mapping != "All":
        filtered = filtered[filtered['Mapping Name'] == selected_mapping]
    if selected_rule != "All":
        filtered = filtered[filtered['Rule Type'] == selected_rule]
    if selected_status != "All":
        filtered = filtered[filtered['Status'] == selected_status]
    
    st.dataframe(filtered, use_container_width=True, height=400)
    
    # Statistics
    st.subheader("Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mappings", df['Mapping Name'].nunique())
    col2.metric("Total Columns", len(df))
    col3.metric("OK", len(df[df['Status'] == 'OK']))
    col4.metric("Exceptions", len(exceptions))
    
    # Rule type chart
    st.subheader("Rule Type Distribution")
    st.bar_chart(df['Rule Type'].value_counts())
    
    # Exceptions report
    st.subheader("Exceptions Report")
    st.text_area("Report", report, height=250)
    
    # Downloads
    st.subheader("Downloads")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "⬇️ Mapping Document (CSV)",
            df.to_csv(index=False),
            f"mapping_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    with col2:
        st.download_button(
            "⬇️ Exceptions Report (TXT)",
            report,
            f"exceptions_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )
    with col3:
        st.download_button(
            "⬇️ All (ZIP)",
            create_download_bundle(df, report, f"informatica_{datetime.now().strftime('%Y%m%d')}"),
            f"informatica_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            "application/zip"
        )

# -- Main Application --

def main():
    """Main application entry point."""
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    st.title("Informatica Mapping Document Generator")
    st.markdown("Generate mapping documentation with **plain English transformation logic** from PowerCenter XML exports.")
    st.divider()
    
    render_sidebar()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("API key not found. Please add it to your `.env` file.")
        return
    
    # File upload
    st.subheader("Upload XML File")
    uploaded_file = st.file_uploader("Choose Informatica PowerCenter XML export", type=['xml'])
    
    if not uploaded_file:
        st.info("Upload an Informatica PowerCenter XML file to get started.")
        return
    
    # Parse XML
    try:
        xml_content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        try:
            xml_content = uploaded_file.read().decode('latin-1')
        except:
            uploaded_file.seek(0)
            xml_content = uploaded_file.read().decode('iso-8859-1')
    
    with st.spinner("Parsing XML..."):
        parsed_data, error = parse_xml(xml_content)
    
    if error:
        st.error(f"Error: {error}")
        return
    
    st.success("XML parsed successfully!")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Sources", len(parsed_data['sources']))
    col2.metric("Targets", len(parsed_data['targets']))
    col3.metric("Mappings", len(parsed_data['mappings']))
    
    # Mapping details
    st.subheader("Mappings Found")
    render_mapping_expanders(parsed_data)
    
    st.divider()
    
    # Generate button
    st.subheader("Generate Documentation")
    
    if st.button("Generate Mapping Document", type="primary"):
        progress = st.progress(0, text="Starting...")
        
        def update_progress(pct, msg):
            progress.progress(pct, text=msg)
        
        with st.spinner("Generating..."):
            df, exceptions = generate_mapping_document(parsed_data, progress_callback=update_progress)
            report = generate_exceptions_report(exceptions, parsed_data)
        
        progress.progress(1.0, text="Complete!")
        st.session_state.results = {'df': df, 'report': report, 'exceptions': exceptions}
        st.success(f"Generated {len(df)} column mappings!")
    
    # Show results
    if st.session_state.results:
        st.divider()
        render_results(
            st.session_state.results['df'],
            st.session_state.results['report'],
            st.session_state.results['exceptions']
        )


if __name__ == "__main__":
    main()